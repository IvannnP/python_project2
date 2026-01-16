
# bot.py
# Telegram bot: water + calories goals, food/workout/water logging, progress report
# aiogram v3.x, in-memory storage (no DB)
#
# Food calories source:
#   - Primary: ChatGPT API (kcal per 100g as a number)
#   - Fallback: OpenFoodFacts (kcal/kJ per 100g)
#
# Workout calories source:
#   - Primary: ChatGPT API (kcal burned for free-form activity + minutes + user weight)
#   - If OpenAI is unavailable -> we do NOT guess; we ask to set OPENAI_API_KEY.
#
# ENV vars:
#   BOT_TOKEN=...
#   OWM_API_KEY=...          (optional)
#   OPENAI_API_KEY=...       (recommended)

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

import requests
from aiogram import BaseMiddleware, Bot, Dispatcher, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Message, TelegramObject
from dotenv import load_dotenv

# OpenAI is optional (bot can run without it, but workout will require it)
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


# -----------------------
# ENV
# -----------------------
env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
OWM_API_KEY = os.getenv("OWM_API_KEY")  # OpenWeatherMap key (optional)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # ChatGPT API key (recommended)

if not BOT_TOKEN:
    raise RuntimeError("No BOT_TOKEN in environment. Put it in .env or export it before running.")


# -----------------------
# LOGGING
# -----------------------
Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/bot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# -----------------------
# DATA (in-memory)
# -----------------------
@dataclass
class DailyLog:
    water_ml: int = 0
    food_kcal: float = 0.0
    burned_kcal: float = 0.0


@dataclass
class UserProfile:
    weight_kg: float
    height_cm: float
    age: int
    sex: str  # "m" or "f"
    activity_min_per_day: int
    city: str
    calorie_goal: Optional[int] = None  # if user set manually

    # computed goals cached daily
    water_goal_ml: int = 0
    calorie_goal_final: int = 0
    last_goal_date: Optional[date] = None

    # per-day logs
    logs_by_date: Dict[date, DailyLog] = field(default_factory=dict)


users: Dict[int, UserProfile] = {}


# -----------------------
# FSM
# -----------------------
class ProfileFSM(StatesGroup):
    weight = State()
    height = State()
    age = State()
    sex = State()
    activity = State()
    city = State()
    calorie_goal = State()


class FoodFSM(StatesGroup):
    waiting_grams = State()


# -----------------------
# Middleware: log every user message (for deployment logs/screenshots)
# -----------------------
class CommandsLoggingMiddleware(BaseMiddleware):
    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict], Awaitable[Any]],
        event: TelegramObject,
        data: dict,
    ) -> Any:
        upd = data.get("event_update")
        msg = getattr(upd, "message", None) if upd else None
        if msg and getattr(msg, "text", None):
            uid = msg.from_user.id if msg.from_user else "unknown"
            logger.info(f"USER {uid}: {msg.text}")
        return await handler(event, data)


# -----------------------
# Helpers: parsing
# -----------------------
def safe_float(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", "."))
    except Exception:
        return None


def safe_int(s: str) -> Optional[int]:
    try:
        return int(s.strip())
    except Exception:
        return None


def fmt_int(n: float) -> str:
    return str(int(round(n)))


def extract_first_number(text: str) -> Optional[float]:
    m = re.search(r"(\d+(?:[.,]\d+)?)", text)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", "."))
    except Exception:
        return None


# -----------------------
# APIs: OpenWeather + OpenFoodFacts + OpenAI
# -----------------------
OFF_BASE = "https://world.openfoodfacts.org"
OFF_HEADERS = {"User-Agent": "WaterFitBot/1.0 (student)"}

openai_client = None
if OpenAI is not None and OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        openai_client = None


def get_temperature_c(city: str) -> Optional[float]:
    """Returns temperature in Celsius from OpenWeatherMap current weather."""
    if not OWM_API_KEY:
        return None
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        r = requests.get(
            url,
            params={"q": city, "appid": OWM_API_KEY, "units": "metric"},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        return float(data["main"]["temp"])
    except Exception:
        return None


def off_search_kcal_100g(query: str) -> Optional[Tuple[str, float]]:
    """Fallback: OpenFoodFacts search -> (product_name, kcal_per_100g)."""
    try:
        url = f"{OFF_BASE}/cgi/search.pl"
        r = requests.get(
            url,
            params={
                "action": "process",
                "search_terms": query,
                "json": "true",
                "page_size": 20,
            },
            headers=OFF_HEADERS,
            timeout=10,
        )
        if r.status_code != 200:
            return None

        data = r.json()
        products = data.get("products", [])
        if not products:
            return None

        for p in products:
            name = (p.get("product_name") or "").strip() or query
            nutr = p.get("nutriments", {}) or {}

            kcal = nutr.get("energy-kcal_100g")
            if kcal is not None:
                try:
                    val = float(kcal)
                    if 0 < val <= 1000:
                        return name, round(val, 1)
                except Exception:
                    pass

            kj = nutr.get("energy_100g")
            if kj is not None:
                try:
                    kcal_from_kj = float(kj) / 4.184
                    if 0 < kcal_from_kj <= 1000:
                        return name, round(kcal_from_kj, 1)
                except Exception:
                    pass

        return None
    except Exception:
        return None


def llm_kcal_100g(query: str) -> Optional[Tuple[str, float]]:
    """ChatGPT: kcal per 100g. Must return only a number; we still extract safely."""
    if not openai_client:
        return None

    try:
        prompt = (
            "Верни оценку калорийности продукта в ккал на 100 грамм.\n"
            "Правила:\n"
            "1) Верни ТОЛЬКО число (например: 89). Без текста и единиц.\n"
            "2) Если это напиток, оцени на 100 грамм.\n"
            "3) Если запрос неоднозначный, выбери самый типичный вариант.\n"
            f"Продукт: {query}"
        )

        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        text = (resp.choices[0].message.content or "").strip()
        kcal = extract_first_number(text)

        if kcal is None or kcal <= 0 or kcal > 1000:
            return None

        return query.strip(), round(float(kcal), 1)
    except Exception as e:
        logger.exception("LLM food kcal error: %s", e)
        return None


def get_kcal_100g(query: str) -> Optional[Tuple[str, float, str]]:
    """Returns (name, kcal_100g, source) where source is 'openai' or 'off'."""
    info = llm_kcal_100g(query)
    if info:
        name, kcal = info
        return name, kcal, "openai"

    info2 = off_search_kcal_100g(query)
    if info2:
        name, kcal = info2
        return name, kcal, "off"

    return None


def llm_workout_kcal(workout_text: str, minutes: int, weight_kg: float) -> Optional[float]:
    """
    ChatGPT: burned kcal for free-form workout text + minutes + weight.
    Returns kcal as number. We extract first number as safety.
    """
    if not openai_client:
        return None

    try:
        prompt = (
            "Оцени количество потраченных калорий (ккал) за тренировку.\n"
            "Дай приблизительную оценку.\n"
            "Правила:\n"
            "1) Верни ТОЛЬКО число (например: 350). Без текста и единиц.\n"
            "2) Учитывай вес человека, длительность и тип тренировки.\n"
            "3) Если интенсивность не указана, считай средней.\n"
            "4) Если запрос странный, всё равно верни реалистичное число.\n"
            f"Вес: {weight_kg} кг\n"
            f"Длительность: {minutes} минут\n"
            f"Тренировка: {workout_text}"
        )

        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        text = (resp.choices[0].message.content or "").strip()
        kcal = extract_first_number(text)

        if kcal is None or kcal <= 0 or kcal > 5000:
            return None
        return round(float(kcal), 1)
    except Exception as e:
        logger.exception("LLM workout kcal error: %s", e)
        return None


# -----------------------
# Helpers: calculations
# -----------------------
def calculate_water_goal_ml(weight_kg: float, activity_min: int, temp_c: Optional[float]) -> int:
    goal = int(weight_kg * 30)  # 30 ml / kg
    goal += int((activity_min / 30) * 500)  # +500 ml per 30 min activity
    if temp_c is not None and temp_c > 25:
        goal += 750
    return goal


def calculate_calorie_goal(weight_kg: float, height_cm: float, age: int, sex: str, activity_min: int) -> int:
    base = 10 * weight_kg + 6.25 * height_cm - 5 * age
    base += 5 if sex == "m" else -161

    if activity_min <= 0:
        add = 0
    elif activity_min <= 30:
        add = 200
    elif activity_min <= 60:
        add = 400
    else:
        add = 450

    return int(round(base + add))


def get_today_log(profile: UserProfile) -> DailyLog:
    today = date.today()
    if today not in profile.logs_by_date:
        profile.logs_by_date[today] = DailyLog()
    return profile.logs_by_date[today]


def ensure_daily_goals(profile: UserProfile) -> None:
    today = date.today()
    if profile.last_goal_date == today:
        return

    temp = get_temperature_c(profile.city)
    profile.water_goal_ml = calculate_water_goal_ml(profile.weight_kg, profile.activity_min_per_day, temp)

    calc_kcal = calculate_calorie_goal(
        profile.weight_kg, profile.height_cm, profile.age, profile.sex, profile.activity_min_per_day
    )
    profile.calorie_goal_final = int(profile.calorie_goal if profile.calorie_goal is not None else calc_kcal)
    profile.last_goal_date = today


def extra_workout_water_ml(minutes: int) -> int:
    # +200 ml per 30 min workout
    return int((max(minutes, 0) / 30) * 200)


# -----------------------
# Router
# -----------------------
router = Router()


@router.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        "👋 Привет! Я помогу считать норму воды и калорий, и вести трекинг.\n\n"
        "Команды:\n"
        "/set_profile — настроить профиль\n"
        "/log_water <мл> — записать воду\n"
        "/log_food <продукт> — записать еду\n"
        "/log_workout <описание> <мин> — записать тренировку (свободный ввод)\n"
        "/check_progress — прогресс за сегодня\n"
        "/help — помощь"
    )


@router.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer(
        "🧭 Команды:\n"
        "• /set_profile\n"
        "• /log_water 300\n"
        "• /log_food банан\n"
        "• /log_workout табата 25\n"
        "• /log_workout футбол 60\n"
        "• /check_progress\n\n"
        "Еда: ChatGPT API → fallback OpenFoodFacts.\n"
        "Тренировки: ChatGPT API (свободный ввод)."
    )


# -----------------------
# Profile setup flow
# -----------------------
@router.message(Command("set_profile"))
async def set_profile(message: Message, state: FSMContext):
    await state.clear()
    await state.set_state(ProfileFSM.weight)
    await message.answer("Введите ваш вес (кг), например: 80")


@router.message(ProfileFSM.weight)
async def profile_weight(message: Message, state: FSMContext):
    v = safe_float(message.text or "")
    if v is None or v <= 0 or v > 400:
        await message.answer("❌ Вес должен быть числом (кг). Например: 80")
        return
    await state.update_data(weight_kg=v)
    await state.set_state(ProfileFSM.height)
    await message.answer("Введите ваш рост (см), например: 184")


@router.message(ProfileFSM.height)
async def profile_height(message: Message, state: FSMContext):
    v = safe_float(message.text or "")
    if v is None or v < 50 or v > 260:
        await message.answer("❌ Рост должен быть числом (см). Например: 184")
        return
    await state.update_data(height_cm=v)
    await state.set_state(ProfileFSM.age)
    await message.answer("Введите ваш возраст, например: 26")


@router.message(ProfileFSM.age)
async def profile_age(message: Message, state: FSMContext):
    v = safe_int(message.text or "")
    if v is None or v < 5 or v > 120:
        await message.answer("❌ Возраст должен быть числом. Например: 26")
        return
    await state.update_data(age=v)
    await state.set_state(ProfileFSM.sex)
    await message.answer("Укажите пол: m (муж) или f (жен)")


@router.message(ProfileFSM.sex)
async def profile_sex(message: Message, state: FSMContext):
    s = (message.text or "").strip().lower()
    if s not in ("m", "f"):
        await message.answer("❌ Введите m или f")
        return
    await state.update_data(sex=s)
    await state.set_state(ProfileFSM.activity)
    await message.answer("Сколько минут активности в день? (например: 45)")


@router.message(ProfileFSM.activity)
async def profile_activity(message: Message, state: FSMContext):
    v = safe_int(message.text or "")
    if v is None or v < 0 or v > 600:
        await message.answer("❌ Минуты активности должны быть числом. Например: 45")
        return
    await state.update_data(activity_min_per_day=v)
    await state.set_state(ProfileFSM.city)
    await message.answer("В каком городе вы находитесь? (например: Moscow)")


@router.message(ProfileFSM.city)
async def profile_city(message: Message, state: FSMContext):
    city = (message.text or "").strip()
    if not city or len(city) < 2:
        await message.answer("❌ Введите город (например: Moscow)")
        return
    await state.update_data(city=city)
    await state.set_state(ProfileFSM.calorie_goal)
    await message.answer(
        "Хотите задать цель калорий вручную?\n"
        "Введите число (например 2500) или напишите 'auto' чтобы рассчитать автоматически."
    )


@router.message(ProfileFSM.calorie_goal)
async def profile_calorie_goal(message: Message, state: FSMContext):
    txt = (message.text or "").strip().lower()
    data = await state.get_data()

    manual_goal: Optional[int] = None
    if txt != "auto":
        v = safe_int(txt)
        if v is None or v < 800 or v > 6000:
            await message.answer("❌ Введите число (например 2500) или 'auto'")
            return
        manual_goal = v

    profile = UserProfile(
        weight_kg=float(data["weight_kg"]),
        height_cm=float(data["height_cm"]),
        age=int(data["age"]),
        sex=str(data["sex"]),
        activity_min_per_day=int(data["activity_min_per_day"]),
        city=str(data["city"]),
        calorie_goal=manual_goal,
    )
    users[message.from_user.id] = profile
    ensure_daily_goals(profile)
    await state.clear()

    temp = get_temperature_c(profile.city)
    temp_txt = f"{temp:.1f}°C" if temp is not None else "не удалось получить (нет OWM_API_KEY или ошибка)"

    gpt_status = "включён" if openai_client else "выключен (нет OPENAI_API_KEY)"
    await message.answer(
        "✅ Профиль сохранён!\n\n"
        f"Город: {profile.city} (температура: {temp_txt})\n"
        f"Норма воды: {profile.water_goal_ml} мл\n"
        f"Цель калорий: {profile.calorie_goal_final} ккал\n"
        f"GPT: {gpt_status}\n\n"
        "Теперь можно логировать:\n"
        "/log_water 300\n"
        "/log_food банан\n"
        "/log_workout табата 25\n"
        "/check_progress"
    )


# -----------------------
# Water logging
# -----------------------
@router.message(Command("log_water"))
async def log_water(message: Message):
    user_id = message.from_user.id
    if user_id not in users:
        await message.answer("Сначала настрой профиль: /set_profile")
        return

    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2:
        await message.answer("Формат: /log_water <мл> (пример: /log_water 300)")
        return

    ml = safe_int(parts[1])
    if ml is None or ml <= 0 or ml > 5000:
        await message.answer("❌ Введите корректное число мл. Например: /log_water 300")
        return

    profile = users[user_id]
    ensure_daily_goals(profile)
    log = get_today_log(profile)

    log.water_ml += ml
    left = max(profile.water_goal_ml - log.water_ml, 0)

    await message.answer(
        f"💧 Записано: {ml} мл.\n"
        f"Сегодня выпито: {log.water_ml} / {profile.water_goal_ml} мл.\n"
        f"Осталось: {left} мл."
    )


# -----------------------
# Food logging (2-step)
# -----------------------
@router.message(Command("log_food"))
async def log_food(message: Message, state: FSMContext):
    user_id = message.from_user.id
    if user_id not in users:
        await message.answer("Сначала настрой профиль: /set_profile")
        return

    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2:
        await message.answer("Формат: /log_food <продукт> (пример: /log_food банан)")
        return

    query = parts[1].strip()
    info = get_kcal_100g(query)
    if not info:
        await message.answer(
            "❌ Не смог определить калорийность.\n"
            "Попробуйте уточнить запрос (например: 'банан', 'гречка варёная', 'капучино без сахара')."
        )
        return

    name, kcal_100g, source = info
    await state.set_state(FoodFSM.waiting_grams)
    await state.update_data(food_name=name, kcal_100g=kcal_100g, source=source)

    src_txt = "ChatGPT" if source == "openai" else "OpenFoodFacts"
    await message.answer(
        f"🍽 {name} — {kcal_100g} ккал на 100 г. (источник: {src_txt})\n"
        "Сколько грамм вы съели? (например: 150)"
    )


@router.message(FoodFSM.waiting_grams)
async def food_grams(message: Message, state: FSMContext):
    user_id = message.from_user.id
    if user_id not in users:
        await state.clear()
        await message.answer("Сначала настрой профиль: /set_profile")
        return

    grams = safe_float(message.text or "")
    if grams is None or grams <= 0 or grams > 5000:
        await message.answer("❌ Введите граммы числом. Например: 150")
        return

    data = await state.get_data()
    name = str(data["food_name"])
    kcal_100g = float(data["kcal_100g"])
    kcal = round(kcal_100g * (grams / 100.0), 1)

    profile = users[user_id]
    ensure_daily_goals(profile)
    log = get_today_log(profile)
    log.food_kcal += kcal

    left = max(profile.calorie_goal_final - log.food_kcal, 0)

    await state.clear()
    await message.answer(
        f"✅ Записано: {name} — {kcal} ккал.\n"
        f"Сегодня съедено: {round(log.food_kcal, 1)} / {profile.calorie_goal_final} ккал.\n"
        f"Осталось до цели: {fmt_int(left)} ккал."
    )


# -----------------------
# Workout logging (FREE FORM + GPT)
# -----------------------
@router.message(Command("log_workout"))
async def log_workout(message: Message):
    user_id = message.from_user.id
    if user_id not in users:
        await message.answer("Сначала настрой профиль: /set_profile")
        return

    # We accept:
    #   /log_workout табата 25
    #   /log_workout футбол 60
    # Parse: last token = minutes, the rest = free-form workout text
    parts = (message.text or "").split()
    if len(parts) < 3:
        await message.answer(
            "Формат: /log_workout <описание> <мин>\n"
            "Примеры:\n"
            "/log_workout табата 25\n"
            "/log_workout футбол 60"
        )
        return

    minutes = safe_int(parts[-1])
    if minutes is None or minutes <= 0 or minutes > 1000:
        await message.answer("❌ Последний аргумент должен быть минутами (число). Пример: /log_workout футбол 60")
        return

    workout_text = " ".join(parts[1:-1]).strip()
    if not workout_text:
        await message.answer("❌ Укажите описание тренировки. Пример: /log_workout силовая тренировка 45")
        return

    profile = users[user_id]
    ensure_daily_goals(profile)
    log = get_today_log(profile)

    burned = llm_workout_kcal(workout_text, minutes, profile.weight_kg)
    if burned is None:
        await message.answer(
            "❌ Не смог посчитать калории тренировки.\n"
            "Проверь, что задан OPENAI_API_KEY (ChatGPT API) и интернет доступен."
        )
        return

    log.burned_kcal += burned
    extra_water = extra_workout_water_ml(minutes)

    await message.answer(
        f"🏋️ Тренировка: {workout_text}\n"
        f"⏱ {minutes} мин — сожжено ~{burned} ккал.\n"
        f"💧 Рекомендация: дополнительно выпейте {extra_water} мл воды."
    )


# -----------------------
# Progress
# -----------------------
@router.message(Command("check_progress"))
async def check_progress(message: Message):
    user_id = message.from_user.id
    if user_id not in users:
        await message.answer("Сначала настрой профиль: /set_profile")
        return

    profile = users[user_id]
    ensure_daily_goals(profile)
    log = get_today_log(profile)

    water_left = max(profile.water_goal_ml - log.water_ml, 0)
    eaten = round(log.food_kcal, 1)
    burned = round(log.burned_kcal, 1)
    balance = round(eaten - burned, 1)

    await message.answer(
        "📊 Прогресс за сегодня:\n\n"
        "💧 Вода:\n"
        f"• Выпито: {log.water_ml} мл из {profile.water_goal_ml} мл\n"
        f"• Осталось: {water_left} мл\n\n"
        "🔥 Калории:\n"
        f"• Потреблено: {eaten} ккал из {profile.calorie_goal_final} ккал\n"
        f"• Сожжено: {burned} ккал\n"
        f"• Баланс (потреблено - сожжено): {balance} ккал"
    )


# -----------------------
# Optional: debug
# -----------------------
@router.message(Command("where_token"))
async def where_token(message: Message):
    exists = "YES" if os.getenv("BOT_TOKEN") else "NO"
    gpt = "YES" if openai_client else "NO"
    await message.answer(f"BOT_TOKEN env present: {exists}\nOPENAI enabled: {gpt}")


@router.message()
async def log_all_messages(message: Message):
    # if no handler matched
    logger.info(f"UNHANDLED USER {message.from_user.id}: {message.text!r}")
    if message.text and not message.text.startswith("/"):
        await message.answer("Я понимаю команды. Напишите /help")


async def main():
    bot = Bot(BOT_TOKEN)  # no parse_mode -> avoids HTML entity errors
    dp = Dispatcher(storage=MemoryStorage())
    dp.update.middleware(CommandsLoggingMiddleware())  # log all updates
    dp.include_router(router)

    logger.info("Bot started.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())