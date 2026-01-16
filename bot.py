# bot.py
# Telegram bot: water + calories goals, food/workout/water logging, progress report
# aiogram v3.x, in-memory storage (no DB)

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests
from aiogram import Bot, Dispatcher, Router, F
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Message
from dotenv import load_dotenv

# -----------------------
# ENV
# -----------------------
# If .env exists near bot.py -> load it. If not, still works with exported env vars.
env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()  # harmless fallback

BOT_TOKEN = os.getenv("BOT_TOKEN")
OWM_API_KEY = os.getenv("OWM_API_KEY")  # OpenWeatherMap key (optional but recommended)

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
# FSM: profile setup
# -----------------------
class ProfileFSM(StatesGroup):
    weight = State()
    height = State()
    age = State()
    sex = State()
    activity = State()
    city = State()
    calorie_goal = State()


# -----------------------
# Helpers: APIs
# -----------------------
OFF_BASE = "https://world.openfoodfacts.org"
OFF_HEADERS = {"User-Agent": "WaterFitBot/1.0 (student)"}


def get_temperature_c(city: str) -> Optional[float]:
    """
    Returns temperature in Celsius from OpenWeatherMap current weather.
    Requires OWM_API_KEY. If no key or error -> None.
    """
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
    """
    Search product in OpenFoodFacts and return (product_name, kcal_per_100g).
    Handles both kcal and kJ properly.
    """
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

            # 1️⃣ Ищем kcal напрямую
            kcal = nutr.get("energy-kcal_100g")
            if kcal is not None:
                try:
                    return name, float(kcal)
                except Exception:
                    pass

            # 2️⃣ Иначе берём kJ и конвертируем
            kj = nutr.get("energy_100g")
            if kj is not None:
                try:
                    kcal_from_kj = float(kj) / 4.184
                    return name, round(kcal_from_kj, 1)
                except Exception:
                    pass

        return None
    except Exception:
        return None

# -----------------------
# Helpers: calculations
# -----------------------
def calculate_water_goal_ml(weight_kg: float, activity_min: int, temp_c: Optional[float]) -> int:
    # base: 30 ml / kg
    goal = int(weight_kg * 30)

    # +500 ml per 30 min activity
    goal += int((activity_min / 30) * 500)

    # +500..1000 ml if hot
    if temp_c is not None and temp_c > 25:
        goal += 750  # middle value for simplicity

    return goal


def calculate_calorie_goal(weight_kg: float, height_cm: float, age: int, sex: str, activity_min: int) -> int:
    """
    Simple Mifflin-St Jeor-like baseline without +5/-161 (we’ll keep close to your spec),
    then add activity calories.
    """
    base = 10 * weight_kg + 6.25 * height_cm - 5 * age
    # sex adjustment (optional, but improves realism)
    if sex == "m":
        base += 5
    else:
        base -= 161

    # activity add: 200..400 depending on minutes (simple linear clamp)
    # 0 min => 0; 30 => 200; 60 => 400; >60 => 450
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
        profile.weight_kg,
        profile.height_cm,
        profile.age,
        profile.sex,
        profile.activity_min_per_day,
    )
    profile.calorie_goal_final = int(profile.calorie_goal if profile.calorie_goal is not None else calc_kcal)
    profile.last_goal_date = today


# -----------------------
# Workout calories (very simple MET-like table)
# -----------------------
WORKOUT_KCAL_PER_MIN = {
    "бег": 10.0,
    "run": 10.0,
    "ходьба": 4.0,
    "walk": 4.0,
    "силовая": 6.0,
    "gym": 6.0,
    "велосипед": 8.0,
    "bike": 8.0,
    "плавание": 9.0,
    "swim": 9.0,
    "йога": 3.0,
    "yoga": 3.0,
}

def estimate_workout_kcal(workout_type: str, minutes: int) -> float:
    key = workout_type.strip().lower()
    rate = WORKOUT_KCAL_PER_MIN.get(key, 6.0)  # default moderate
    return round(rate * max(minutes, 0), 1)

def extra_workout_water_ml(minutes: int) -> int:
    # +200 ml per 30 min workout
    return int((max(minutes, 0) / 30) * 200)


# -----------------------
# Bot / Router
# -----------------------
router = Router()

def fmt_int(n: float) -> str:
    return str(int(round(n)))

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


@router.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        "👋 Привет! Я помогу считать норму воды и калорий, и вести трекинг.\n\n"
        "Команды:\n"
        "/set_profile — настроить профиль\n"
        "/log_water <мл> — записать воду\n"
        "/log_food <продукт> — записать еду\n"
        "/log_workout <тип> <мин> — записать тренировку\n"
        "/check_progress — прогресс за сегодня\n"
        "/help — помощь",
    )


@router.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer(
        "🧭 Команды:\n"
        "• /set_profile\n"
        "• /log_water 300\n"
        "• /log_food banana\n"
        "• /log_workout бег 30\n"
        "• /check_progress\n\n"
        "Совет: OpenFoodFacts лучше ищет по-английски (banana, apple, bread).",
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

    await message.answer(
        "✅ Профиль сохранён!\n\n"
        f"Город: {profile.city} (температура: {temp_txt})\n"
        f"Норма воды: {profile.water_goal_ml} мл\n"
        f"Цель калорий: {profile.calorie_goal_final} ккал\n\n"
        "Теперь можно логировать:\n"
        "/log_water 300\n"
        "/log_food banana\n"
        "/log_workout бег 30\n"
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
# Food logging (2-step): /log_food name -> ask grams -> calculate
# -----------------------
class FoodFSM(StatesGroup):
    waiting_grams = State()


@router.message(Command("log_food"))
async def log_food(message: Message, state: FSMContext):
    user_id = message.from_user.id
    if user_id not in users:
        await message.answer("Сначала настрой профиль: /set_profile")
        return

    parts = (message.text or "").split(maxsplit=1)
    if len(parts) < 2:
        await message.answer("Формат: /log_food <продукт> (пример: /log_food banana)")
        return

    query = parts[1].strip()
    info = off_search_kcal_100g(query)

    # fallback: if Russian query often fails, try same as-is (we already did),
    # but suggest using English if not found.
    if not info:
        await message.answer(
            "❌ Не смог найти продукт в OpenFoodFacts.\n"
            "Попробуйте на английском (например: banana, apple, bread)."
        )
        return

    name, kcal_100g = info
    await state.set_state(FoodFSM.waiting_grams)
    await state.update_data(food_name=name, kcal_100g=kcal_100g)

    await message.answer(f"🍽 {name} — {kcal_100g} ккал на 100 г.\nСколько грамм вы съели? (например: 150)")


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
        f"Сегодня съедено: {round(log.food_kcal,1)} / {profile.calorie_goal_final} ккал.\n"
        f"Осталось до цели: {fmt_int(left)} ккал."
    )


# -----------------------
# Workout logging
# -----------------------
@router.message(Command("log_workout"))
async def log_workout(message: Message):
    user_id = message.from_user.id
    if user_id not in users:
        await message.answer("Сначала настрой профиль: /set_profile")
        return

    parts = (message.text or "").split()
    if len(parts) < 3:
        await message.answer("Формат: /log_workout <тип> <мин>\nПример: /log_workout бег 30")
        return

    workout_type = parts[1]
    minutes = safe_int(parts[2])
    if minutes is None or minutes <= 0 or minutes > 1000:
        await message.answer("❌ Минуты должны быть числом. Пример: /log_workout бег 30")
        return

    profile = users[user_id]
    ensure_daily_goals(profile)
    log = get_today_log(profile)

    burned = estimate_workout_kcal(workout_type, minutes)
    log.burned_kcal += burned

    extra_water = extra_workout_water_ml(minutes)
    # We do NOT auto-add water to drunk water. It's a recommendation.
    await message.answer(
        f"🏋️ {workout_type} {minutes} мин — сожжено ~{burned} ккал.\n"
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
    balance = round(eaten - burned, 1)  # net intake

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
# Optional: debug command
# -----------------------
@router.message(Command("where_token"))
async def where_token(message: Message):
    # For local debug only. Doesn't print token. Shows if env var exists.
    exists = "YES" if os.getenv("BOT_TOKEN") else "NO"
    await message.answer(f"BOT_TOKEN env present: {exists}")


# -----------------------
# Middleware-like logging (simple)
# -----------------------
@router.message()
async def log_all_messages(message: Message):
    # This handler triggers only if no previous handler matched.
    # Helpful to see that the bot receives messages even if command doesn't parse.
    logger.info(f"UNHANDLED USER {message.from_user.id}: {message.text!r}")
    # Do not spam user; just hint once:
    if message.text and not message.text.startswith("/"):
        await message.answer("Я понимаю команды. Напишите /help")


async def main():
    bot = Bot(BOT_TOKEN)
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)

    logger.info("Bot started.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())