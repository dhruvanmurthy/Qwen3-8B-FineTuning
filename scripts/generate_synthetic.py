"""
Generate synthetic tool-use training examples.

Creates realistic function-calling / API-orchestration examples that cover:
 - Single-tool calls
 - Multi-step tool chains
 - Argument type diversity (string, int, float, bool, list, null)
 - Error / fallback scenarios

Output: data/raw/synthetic/synthetic.jsonl
        data/raw/synthetic/synthetic_multistep.jsonl

Usage:
    python scripts/generate_synthetic.py [--num-samples 15000] [--seed 42]
"""

import argparse
import json
import random
import sys
from pathlib import Path

# ── Tool catalogue ────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "city": {"type": "string", "required": True},
            "units": {"type": "string", "enum": ["C", "F"], "required": False, "default": "C"},
        },
    },
    {
        "name": "get_stock_price",
        "description": "Get current stock price for a ticker symbol",
        "parameters": {
            "symbol": {"type": "string", "required": True},
            "currency": {"type": "string", "required": False, "default": "USD"},
        },
    },
    {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "query": {"type": "string", "required": True},
            "num_results": {"type": "integer", "required": False, "default": 5},
        },
    },
    {
        "name": "send_email",
        "description": "Send an email to a recipient",
        "parameters": {
            "to": {"type": "string", "required": True},
            "subject": {"type": "string", "required": True},
            "body": {"type": "string", "required": True},
            "cc": {"type": "string", "required": False},
        },
    },
    {
        "name": "create_calendar_event",
        "description": "Create a new calendar event",
        "parameters": {
            "title": {"type": "string", "required": True},
            "date": {"type": "string", "required": True},
            "time": {"type": "string", "required": True},
            "duration_minutes": {"type": "integer", "required": False, "default": 60},
            "attendees": {"type": "array", "required": False, "default": []},
        },
    },
    {
        "name": "translate_text",
        "description": "Translate text from one language to another",
        "parameters": {
            "text": {"type": "string", "required": True},
            "source_lang": {"type": "string", "required": False, "default": "auto"},
            "target_lang": {"type": "string", "required": True},
        },
    },
    {
        "name": "get_news",
        "description": "Get latest news articles on a topic",
        "parameters": {
            "topic": {"type": "string", "required": True},
            "num_articles": {"type": "integer", "required": False, "default": 3},
            "language": {"type": "string", "required": False, "default": "en"},
        },
    },
    {
        "name": "calculate",
        "description": "Perform a mathematical calculation",
        "parameters": {
            "expression": {"type": "string", "required": True},
        },
    },
    {
        "name": "get_restaurant_info",
        "description": "Get information about a restaurant",
        "parameters": {
            "name": {"type": "string", "required": True},
            "city": {"type": "string", "required": True},
        },
    },
    {
        "name": "book_hotel",
        "description": "Book a hotel room",
        "parameters": {
            "hotel_name": {"type": "string", "required": True},
            "check_in": {"type": "string", "required": True},
            "check_out": {"type": "string", "required": True},
            "guests": {"type": "integer", "required": False, "default": 1},
            "room_type": {"type": "string", "required": False, "default": "standard"},
        },
    },
    {
        "name": "get_directions",
        "description": "Get directions between two locations",
        "parameters": {
            "origin": {"type": "string", "required": True},
            "destination": {"type": "string", "required": True},
            "mode": {"type": "string", "enum": ["driving", "walking", "transit"], "required": False, "default": "driving"},
        },
    },
    {
        "name": "convert_units",
        "description": "Convert a value from one unit to another",
        "parameters": {
            "value": {"type": "number", "required": True},
            "from_unit": {"type": "string", "required": True},
            "to_unit": {"type": "string", "required": True},
        },
    },
    {
        "name": "get_exchange_rate",
        "description": "Get the exchange rate between two currencies",
        "parameters": {
            "from_currency": {"type": "string", "required": True},
            "to_currency": {"type": "string", "required": True},
        },
    },
    {
        "name": "set_reminder",
        "description": "Set a reminder for a specific time",
        "parameters": {
            "message": {"type": "string", "required": True},
            "datetime": {"type": "string", "required": True},
            "repeat": {"type": "string", "enum": ["once", "daily", "weekly"], "required": False, "default": "once"},
        },
    },
    {
        "name": "play_music",
        "description": "Play music by artist, song, or playlist",
        "parameters": {
            "query": {"type": "string", "required": True},
            "source": {"type": "string", "enum": ["spotify", "youtube", "local"], "required": False, "default": "spotify"},
        },
    },
]

CITIES = [
    "New York", "London", "Tokyo", "Paris", "Sydney", "Toronto", "Berlin",
    "Singapore", "Dubai", "Mumbai", "São Paulo", "Seoul", "Amsterdam", "Chicago",
    "Los Angeles", "Bangkok", "Mexico City", "Cairo", "Istanbul", "Moscow",
    "Shanghai", "Beijing", "Jakarta", "Lagos", "Nairobi", "Cape Town", "Bogotá",
    "Buenos Aires", "Lima", "Karachi", "Manila", "Kuala Lumpur", "Riyadh",
    "Casablanca", "Johannesburg", "Lahore", "Hyderabad", "Chennai", "Kolkata",
    "Bangalore", "Pune", "Ahmedabad", "Vienna", "Zürich", "Brussels",
    "Stockholm", "Oslo", "Helsinki", "Copenhagen", "Warsaw", "Prague",
    "Budapest", "Athens", "Lisbon", "Madrid", "Barcelona", "Milan", "Rome",
    "Frankfurt", "Hamburg", "Munich", "Melbourne", "Brisbane", "Perth",
    "Auckland", "San Francisco", "Seattle", "Boston", "Miami", "Atlanta",
    "Denver", "Phoenix", "Dallas", "Houston", "San Diego", "Portland",
    "Las Vegas", "Minneapolis", "Montreal", "Vancouver", "Calgary",
]

TICKERS = [
    "AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "AMD", "NFLX",
    "UBER", "LYFT", "SPOT", "SHOP", "SQ", "PYPL", "INTC", "ORCL", "IBM",
    "CRM", "ADBE", "QCOM", "AVGO", "TXN", "TSM", "BABA", "JD", "NIO",
    "WMT", "TGT", "COST", "HD", "JPM", "BAC", "C", "WFC", "GS", "MS",
    "V", "MA", "AXP", "DIS", "CMCSA", "T", "VZ", "TMO", "ABT", "JNJ",
    "PFE", "MRNA", "LLY", "MRK", "GILD", "REGN", "AMGN", "COIN", "HOOD",
    "PLTR", "RBLX", "SNAP", "PINS", "ZM", "DOCU", "TWLO", "NET", "DDOG",
]

CURRENCIES = [
    "USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "CNY", "INR", "BRL",
    "RUB", "KRW", "SGD", "HKD", "NOK", "SEK", "DKK", "MXN", "ZAR", "AED",
    "SAR", "THB", "IDR", "MYR", "PHP", "TWD", "CZK", "PLN", "HUF", "TRY",
]

LANGUAGES = {
    "es": "Spanish", "fr": "French", "de": "German", "ja": "Japanese",
    "zh": "Chinese", "ar": "Arabic", "pt": "Portuguese", "ru": "Russian",
    "ko": "Korean", "hi": "Hindi", "it": "Italian", "nl": "Dutch",
    "sv": "Swedish", "pl": "Polish", "tr": "Turkish", "vi": "Vietnamese",
    "id": "Indonesian", "uk": "Ukrainian", "ro": "Romanian", "el": "Greek",
}

NEWS_TOPICS = [
    "technology", "finance", "sports", "politics", "science", "health",
    "entertainment", "climate change", "artificial intelligence", "cryptocurrency",
    "space exploration", "renewable energy", "electric vehicles", "cybersecurity",
    "quantum computing", "biotechnology", "geopolitics", "education",
    "real estate", "startups", "gaming", "social media", "e-commerce",
    "autonomous vehicles", "mental health",
]

DATE_PATTERNS = [
    "2026-04-01", "2026-04-05", "2026-04-10", "2026-04-15", "2026-04-20",
    "2026-04-25", "2026-05-01", "2026-05-10", "2026-05-15", "2026-05-20",
    "2026-05-25", "2026-06-01", "2026-06-10", "2026-06-15", "2026-06-20",
    "2026-06-25", "2026-07-01", "2026-07-04", "2026-07-10", "2026-07-15",
    "2026-07-20", "2026-08-01", "2026-08-10", "2026-08-15", "2026-09-01",
    "2026-09-10", "2026-10-01", "2026-10-15", "2026-11-01", "2026-12-01",
    "2026-12-25", "2027-01-01", "2027-01-15", "2027-02-14", "2027-03-01",
]

TIME_PATTERNS = [
    "07:00", "07:30", "08:00", "08:30", "09:00", "09:30", "10:00", "10:30",
    "11:00", "11:30", "12:00", "12:30", "13:00", "13:30", "14:00", "14:30",
    "15:00", "15:30", "16:00", "16:30", "17:00", "17:30", "18:00", "18:30",
    "19:00", "19:30", "20:00", "20:30", "21:00", "21:30", "22:00",
]

PEOPLE = [
    "alice@company.com", "bob@work.org", "carol@firm.io", "dave@startup.co",
    "eve@corp.net", "frank@labs.ai", "grace@tech.com", "henry@uni.edu",
    "irene@research.org", "james@consulting.com", "kate@media.co", "liam@bank.com",
    "mia@hospital.org", "noah@legal.net", "olivia@school.edu", "peter@hotel.com",
    "quinn@press.io", "rose@design.co", "sam@agency.net", "tara@health.org",
]

REMINDER_MESSAGES = [
    "Take medication", "Team standup meeting", "Call mom", "Submit weekly report",
    "Pay rent", "Gym session", "Doctor appointment", "Project deadline",
    "Pick up kids from school", "Car service appointment", "Dentist visit",
    "Grocery shopping", "Flight check-in", "Hotel checkout", "Pay credit card bill",
    "Renew subscription", "Birthday dinner reservation", "Conference call",
    "Code review session", "Performance review", "Training session",
    "Submit expense report", "Quarterly board meeting", "Product demo",
    "Client presentation", "System backup", "Tax filing deadline",
    "Passport renewal", "Insurance payment", "Annual health check-up",
]

PHRASES_TO_TRANSLATE = [
    "Hello, how are you today?", "The weather is beautiful outside.",
    "Can you help me find a good restaurant?", "I need directions to the airport.",
    "What time does the meeting start?", "Thank you very much for your help.",
    "I would like to book a table for two.", "Where is the nearest pharmacy?",
    "How much does this cost?", "Could you please speak more slowly?",
    "I don't understand. Can you repeat that?", "Is there a doctor available?",
    "I have lost my passport.", "What is the WiFi password?",
    "I am allergic to nuts.", "Please call me a taxi.",
    "The bill, please.", "Do you accept credit cards?",
    "I need to cancel my reservation.", "When does the next train leave?",
    "How long is the flight?", "I am looking for a vegetarian option.",
    "Can I have a wake-up call at 7am?", "Where is the nearest ATM?",
    "I would like to check out.",
]

MUSIC_QUERIES = [
    "Bohemian Rhapsody by Queen", "Blinding Lights by The Weeknd",
    "Jazz classics playlist", "Lo-fi hip hop beats", "Classical piano Chopin",
    "Taylor Swift greatest hits", "Top 50 Global Spotify", "90s rock anthems",
    "Morning workout playlist", "Relaxing ambient music", "Ed Sheeran Shape of You",
    "Billie Eilish discography", "Daft Punk Random Access Memories",
    "Focus mode deep work playlist", "Dinner party jazz", "Hip hop 2025 hits",
    "Adele Someone Like You", "Coldplay Yellow", "Dance party playlist",
    "Metallica Nothing Else Matters", "Classical music for studying",
]

SEARCH_QUERIES = [
    "best Italian restaurants near me", "how to fix a leaking faucet",
    "Python asyncio tutorial 2026", "best hiking trails in Colorado",
    "home remedies for a cold", "how to invest in index funds",
    "machine learning interview questions", "easiest programming language to learn",
    "electric car charging stations near me", "best noise cancelling headphones 2026",
    "how to meditate for beginners", "vegan protein recipes",
    "remote work productivity tips", "how to negotiate a salary",
    "real estate investment basics", "cloud computing certifications",
    "data science career path", "sustainable living tips",
    "how to start a podcast", "best budget travel destinations 2026",
    "learn Spanish in 30 days", "home gym equipment recommendations",
    "dividend investing strategy", "how to write a cover letter",
    "best project management tools",
]

UNIT_CONVERSIONS = [
    ("km", "miles", lambda r: round(r.uniform(1, 1000), 1)),
    ("miles", "km", lambda r: round(r.uniform(1, 600), 1)),
    ("kg", "lbs", lambda r: round(r.uniform(0.5, 200), 1)),
    ("lbs", "kg", lambda r: round(r.uniform(1, 440), 1)),
    ("celsius", "fahrenheit", lambda r: round(r.uniform(-20, 45), 1)),
    ("fahrenheit", "celsius", lambda r: round(r.uniform(-4, 113), 1)),
    ("meters", "feet", lambda r: round(r.uniform(1, 1000), 1)),
    ("feet", "meters", lambda r: round(r.uniform(1, 3000), 1)),
    ("liters", "gallons", lambda r: round(r.uniform(0.5, 100), 1)),
    ("gallons", "liters", lambda r: round(r.uniform(0.1, 26), 1)),
    ("inches", "cm", lambda r: round(r.uniform(1, 72), 1)),
    ("cm", "inches", lambda r: round(r.uniform(1, 200), 1)),
    ("oz", "grams", lambda r: round(r.uniform(0.5, 64), 1)),
    ("grams", "oz", lambda r: round(r.uniform(1, 1814), 0)),
    ("MB", "GB", lambda r: round(r.uniform(100, 100000), 0)),
    ("mph", "kph", lambda r: round(r.uniform(5, 200), 1)),
    ("kph", "mph", lambda r: round(r.uniform(8, 320), 1)),
    ("acres", "hectares", lambda r: round(r.uniform(0.5, 500), 1)),
    ("hectares", "acres", lambda r: round(r.uniform(0.2, 200), 1)),
    ("calories", "kilojoules", lambda r: round(r.uniform(100, 3000), 0)),
]

MATH_TEMPLATES = [
    ("{a} + {b}", "What is {a} plus {b}?"),
    ("{a} - {b}", "What is {a} minus {b}?"),
    ("{a} * {b}", "What is {a} multiplied by {b}?"),
    ("{a} / {b}", "What is {a} divided by {b}?"),
    ("{a} ** 2", "What is {a} squared?"),
    ("({a} + {b}) * {c}", "Calculate ({a} + {b}) times {c}."),
    ("{a} % {b}", "What is {a} modulo {b}?"),
    ("{a} * {b} + {c}", "What is {a} times {b} plus {c}?"),
    ("round({a} / {b}, 2)", "Divide {a} by {b} and round to 2 decimal places."),
    ("{a} ** 3", "What is {a} cubed?"),
]

EMAIL_SUBJECTS = [
    "Meeting tomorrow", "Project update", "Quick question", "Report attached",
    "Follow-up from our call", "Invitation", "Action required", "FYI",
    "Weekly summary", "Urgent: please review", "Re: your request",
    "New proposal", "Budget review", "Team offsite planning", "Job offer",
]

EMAIL_BODIES = [
    "Hi, please see the attached report.",
    "Hi, just following up on our earlier conversation.",
    "Hello, do you have a moment to discuss this?",
    "Please find the requested information below.",
    "Quick update: everything is on track.",
    "I wanted to share a few thoughts on this topic.",
    "Could you review this at your earliest convenience?",
    "Kindly confirm receipt of this email.",
]

HOTEL_TYPES = ["standard", "deluxe", "suite", "penthouse", "executive", "economy"]


# ── Template-based example generation ─────────────────────────────────────────

def make_weather_example(rng: random.Random) -> dict:
    city = rng.choice(CITIES)
    units = rng.choice(["C", "F"])
    unit_word = "Celsius" if units == "C" else "Fahrenheit"
    instructions = [
        f"What is the weather like in {city}?",
        f"Check the current weather in {city} in {unit_word}.",
        f"Is it raining in {city} right now?",
        f"Tell me the temperature in {city}.",
        f"What should I wear in {city} today?",
        f"How hot or cold is it in {city}?",
        f"Get me the current weather for {city} in {unit_word}.",
        f"What's the weather forecast for {city}?",
        f"Do I need an umbrella in {city} today?",
    ]
    return {
        "instruction": rng.choice(instructions),
        "tools": [t for t in TOOLS if t["name"] == "get_weather"],
        "tool_calls": [{"name": "get_weather", "arguments": {"city": city, "units": units}}],
        "category": "weather",
        "num_steps": 1,
    }


def make_stock_example(rng: random.Random) -> dict:
    ticker = rng.choice(TICKERS)
    currency = rng.choice(["USD", "EUR", "GBP"])
    instructions = [
        f"What is the current price of {ticker}?",
        f"How much is {ticker} trading at today?",
        f"Get me the stock price for {ticker}.",
        f"Check the latest price for {ticker} shares.",
        f"What's {ticker} at right now?",
        f"Fetch the current {ticker} stock value in {currency}.",
        f"Is {ticker} up or down today?",
        f"How is {ticker} performing in the market?",
    ]
    args = {"symbol": ticker}
    if rng.random() > 0.5:
        args["currency"] = currency
    return {
        "instruction": rng.choice(instructions),
        "tools": [t for t in TOOLS if t["name"] == "get_stock_price"],
        "tool_calls": [{"name": "get_stock_price", "arguments": args}],
        "category": "finance",
        "num_steps": 1,
    }


def make_translate_example(rng: random.Random) -> dict:
    target, lang_name = rng.choice(list(LANGUAGES.items()))
    text = rng.choice(PHRASES_TO_TRANSLATE)
    instructions = [
        f"Translate '{text}' to {lang_name}.",
        f"How do you say '{text}' in {lang_name}?",
        f"Convert this to {lang_name}: {text}",
        f"What is the {lang_name} translation of: {text}",
        f"I need '{text}' in {lang_name}.",
        f"Please translate the following to {lang_name}: {text}",
    ]
    return {
        "instruction": rng.choice(instructions),
        "tools": [t for t in TOOLS if t["name"] == "translate_text"],
        "tool_calls": [{"name": "translate_text", "arguments": {"text": text, "target_lang": target}}],
        "category": "translation",
        "num_steps": 1,
    }


def make_calculate_example(rng: random.Random) -> dict:
    expr_template, instr_template = rng.choice(MATH_TEMPLATES)
    a = rng.randint(2, 9999)
    b = rng.randint(2, 999)
    c = rng.randint(2, 99)
    expr = expr_template.format(a=a, b=b, c=c)
    instruction = instr_template.format(a=a, b=b, c=c)
    return {
        "instruction": instruction,
        "tools": [t for t in TOOLS if t["name"] == "calculate"],
        "tool_calls": [{"name": "calculate", "arguments": {"expression": expr}}],
        "category": "math",
        "num_steps": 1,
    }


def make_convert_units_example(rng: random.Random) -> dict:
    from_unit, to_unit, val_fn = rng.choice(UNIT_CONVERSIONS)
    value = val_fn(rng)
    instructions = [
        f"Convert {value} {from_unit} to {to_unit}.",
        f"How many {to_unit} is {value} {from_unit}?",
        f"What is {value} {from_unit} in {to_unit}?",
        f"I need to convert {value} {from_unit} into {to_unit}.",
        f"Express {value} {from_unit} as {to_unit}.",
    ]
    return {
        "instruction": rng.choice(instructions),
        "tools": [t for t in TOOLS if t["name"] == "convert_units"],
        "tool_calls": [{"name": "convert_units", "arguments": {
            "value": value, "from_unit": from_unit, "to_unit": to_unit
        }}],
        "category": "conversion",
        "num_steps": 1,
    }


def make_reminder_example(rng: random.Random) -> dict:
    msg = rng.choice(REMINDER_MESSAGES)
    date = rng.choice(DATE_PATTERNS)
    time = rng.choice(TIME_PATTERNS)
    repeat = rng.choice(["once", "daily", "weekly", None, None])
    instructions = [
        f"Set a reminder to '{msg}' on {date} at {time}.",
        f"Remind me to {msg.lower()} at {time} on {date}.",
        f"Create a reminder for '{msg}' — {date} {time}.",
        f"I need a reminder about '{msg}' scheduled for {date} at {time}.",
        f"Add '{msg}' to my reminders for {date}, {time}.",
    ]
    args = {"message": msg, "datetime": f"{date}T{time}:00"}
    if repeat:
        args["repeat"] = repeat
    return {
        "instruction": rng.choice(instructions),
        "tools": [t for t in TOOLS if t["name"] == "set_reminder"],
        "tool_calls": [{"name": "set_reminder", "arguments": args}],
        "category": "productivity",
        "num_steps": 1,
    }


def make_exchange_rate_example(rng: random.Random) -> dict:
    from_c, to_c = rng.sample(CURRENCIES, 2)
    amount = rng.choice([1, 50, 100, 500, 1000, 5000, 10000])
    instructions = [
        f"What is the exchange rate from {from_c} to {to_c}?",
        f"How much is {amount} {from_c} in {to_c}?",
        f"Get the {from_c}/{to_c} exchange rate.",
        f"Convert {amount} {from_c} to {to_c}.",
        f"What's the current {from_c} to {to_c} rate?",
        f"I need the exchange rate between {from_c} and {to_c}.",
    ]
    return {
        "instruction": rng.choice(instructions),
        "tools": [t for t in TOOLS if t["name"] == "get_exchange_rate"],
        "tool_calls": [{"name": "get_exchange_rate", "arguments": {
            "from_currency": from_c, "to_currency": to_c
        }}],
        "category": "finance",
        "num_steps": 1,
    }


def make_news_example(rng: random.Random) -> dict:
    topic = rng.choice(NEWS_TOPICS)
    n = rng.choice([3, 5, 7, 10])
    instructions = [
        f"Get me the latest news about {topic}.",
        f"What are the top {n} news stories on {topic}?",
        f"Show me recent headlines about {topic}.",
        f"Find the latest {n} articles on {topic}.",
        f"What's happening in {topic} right now?",
        f"Give me a news update on {topic}.",
    ]
    return {
        "instruction": rng.choice(instructions),
        "tools": [t for t in TOOLS if t["name"] == "get_news"],
        "tool_calls": [{"name": "get_news", "arguments": {"topic": topic, "num_articles": n}}],
        "category": "news",
        "num_steps": 1,
    }


def make_search_example(rng: random.Random) -> dict:
    query = rng.choice(SEARCH_QUERIES)
    n = rng.choice([3, 5, 8, 10])
    instructions = [
        f"Search the web for: {query}",
        f"Find information about {query}.",
        f"Look up '{query}' online.",
        f"I need {n} search results for: {query}",
        f"Google: {query}",
        f"Can you search for {query}?",
    ]
    return {
        "instruction": rng.choice(instructions),
        "tools": [t for t in TOOLS if t["name"] == "search_web"],
        "tool_calls": [{"name": "search_web", "arguments": {"query": query, "num_results": n}}],
        "category": "search",
        "num_steps": 1,
    }


def make_music_example(rng: random.Random) -> dict:
    query = rng.choice(MUSIC_QUERIES)
    source = rng.choice(["spotify", "youtube", "local", None])
    instructions = [
        f"Play {query}.",
        f"I want to listen to {query}.",
        f"Put on {query}.",
        f"Start playing {query}.",
        f"Queue up {query}.",
        f"Play some {query} for me.",
    ]
    args = {"query": query}
    if source:
        args["source"] = source
    return {
        "instruction": rng.choice(instructions),
        "tools": [t for t in TOOLS if t["name"] == "play_music"],
        "tool_calls": [{"name": "play_music", "arguments": args}],
        "category": "entertainment",
        "num_steps": 1,
    }


def make_directions_example(rng: random.Random) -> dict:
    origin, dest = rng.sample(CITIES, 2)
    mode = rng.choice(["driving", "walking", "transit", "driving"])  # driving more common
    instructions = [
        f"How do I get from {origin} to {dest}?",
        f"Give me directions from {origin} to {dest}.",
        f"Navigate from {origin} to {dest} by {mode}.",
        f"What's the best route from {origin} to {dest}?",
        f"I need {mode} directions from {origin} to {dest}.",
        f"Get me from {origin} to {dest}.",
    ]
    return {
        "instruction": rng.choice(instructions),
        "tools": [t for t in TOOLS if t["name"] == "get_directions"],
        "tool_calls": [{"name": "get_directions", "arguments": {
            "origin": origin, "destination": dest, "mode": mode
        }}],
        "category": "navigation",
        "num_steps": 1,
    }


def make_send_email_example(rng: random.Random) -> dict:
    to = rng.choice(PEOPLE)
    subject = rng.choice(EMAIL_SUBJECTS)
    body = rng.choice(EMAIL_BODIES)
    instructions = [
        f"Send an email to {to} with subject '{subject}'.",
        f"Email {to}: {subject}",
        f"Write an email to {to} about {subject}.",
        f"Compose and send an email to {to} — Subject: {subject}.",
        f"Send {to} an email titled '{subject}'.",
    ]
    return {
        "instruction": rng.choice(instructions),
        "tools": [t for t in TOOLS if t["name"] == "send_email"],
        "tool_calls": [{"name": "send_email", "arguments": {
            "to": to, "subject": subject, "body": body
        }}],
        "category": "communication",
        "num_steps": 1,
    }


# ── Multi-step generators ─────────────────────────────────────────────────────

def make_multistep_weather_stock(rng: random.Random) -> dict:
    city = rng.choice(CITIES)
    ticker = rng.choice(TICKERS)
    units = rng.choice(["C", "F"])
    instructions = [
        f"Get the current weather in {city} and check the stock price of {ticker}.",
        f"What's the weather in {city} and how is {ticker} doing today?",
        f"I need two things: weather for {city} and the latest price of {ticker}.",
        f"Check {city} weather and {ticker} stock for me.",
    ]
    return {
        "instruction": rng.choice(instructions),
        "tools": [t for t in TOOLS if t["name"] in ("get_weather", "get_stock_price")],
        "tool_calls": [
            {"name": "get_weather", "arguments": {"city": city, "units": units}},
            {"name": "get_stock_price", "arguments": {"symbol": ticker}},
        ],
        "category": "multi_step",
        "num_steps": 2,
    }


def make_multistep_news_email(rng: random.Random) -> dict:
    topic = rng.choice(NEWS_TOPICS)
    recipient = rng.choice(PEOPLE)
    n = rng.choice([3, 5])
    instructions = [
        f"Find the latest {n} news articles about {topic} and send a summary to {recipient}.",
        f"Get top {n} news on {topic}, then email {recipient} with a summary.",
        f"Research {topic} in the news and forward the findings to {recipient}.",
        f"Look up recent {topic} news and email {recipient} about it.",
    ]
    return {
        "instruction": rng.choice(instructions),
        "tools": [t for t in TOOLS if t["name"] in ("get_news", "send_email")],
        "tool_calls": [
            {"name": "get_news", "arguments": {"topic": topic, "num_articles": n}},
            {"name": "send_email", "arguments": {
                "to": recipient,
                "subject": f"Latest news: {topic}",
                "body": f"Here is a summary of the latest {n} news articles on {topic}.",
            }},
        ],
        "category": "multi_step",
        "num_steps": 2,
    }


def make_multistep_directions_hotel(rng: random.Random) -> dict:
    origin, dest = rng.sample(CITIES, 2)
    check_in = rng.choice(DATE_PATTERNS[:15])
    check_out = rng.choice(DATE_PATTERNS[15:])
    guests = rng.randint(1, 4)
    room = rng.choice(HOTEL_TYPES)
    instructions = [
        f"Get driving directions from {origin} to {dest} and book a {room} hotel room there from {check_in} to {check_out}.",
        f"I'm travelling from {origin} to {dest}. Show me directions and reserve a hotel in {dest}.",
        f"Plan my trip: directions from {origin} to {dest}, then book a hotel in {dest} for {guests} guests.",
        f"Navigate from {origin} to {dest} and arrange a {room} room in {dest} ({check_in}–{check_out}).",
    ]
    return {
        "instruction": rng.choice(instructions),
        "tools": [t for t in TOOLS if t["name"] in ("get_directions", "book_hotel")],
        "tool_calls": [
            {"name": "get_directions", "arguments": {"origin": origin, "destination": dest}},
            {"name": "book_hotel", "arguments": {
                "hotel_name": f"Grand Hotel {dest}",
                "check_in": check_in, "check_out": check_out,
                "guests": guests, "room_type": room,
            }},
        ],
        "category": "multi_step",
        "num_steps": 2,
    }


def make_multistep_exchange_convert(rng: random.Random) -> dict:
    from_c, to_c = rng.sample(CURRENCIES[:10], 2)
    from_unit, to_unit, val_fn = rng.choice(UNIT_CONVERSIONS[:8])
    value = val_fn(rng)
    instructions = [
        f"Get the {from_c} to {to_c} exchange rate and also convert {value} {from_unit} to {to_unit}.",
        f"I need two conversions: currency {from_c}/{to_c} and unit {value} {from_unit} to {to_unit}.",
        f"What's {from_c} in {to_c}? Also, convert {value} {from_unit} to {to_unit}.",
    ]
    return {
        "instruction": rng.choice(instructions),
        "tools": [t for t in TOOLS if t["name"] in ("get_exchange_rate", "convert_units")],
        "tool_calls": [
            {"name": "get_exchange_rate", "arguments": {"from_currency": from_c, "to_currency": to_c}},
            {"name": "convert_units", "arguments": {"value": value, "from_unit": from_unit, "to_unit": to_unit}},
        ],
        "category": "multi_step",
        "num_steps": 2,
    }


def make_multistep_search_email(rng: random.Random) -> dict:
    query = rng.choice(SEARCH_QUERIES)
    recipient = rng.choice(PEOPLE)
    n = rng.choice([5, 10])
    instructions = [
        f"Search for '{query}' and email the top {n} results to {recipient}.",
        f"Look up '{query}' online and send a summary to {recipient}.",
        f"Find information about {query} and forward it to {recipient}.",
    ]
    return {
        "instruction": rng.choice(instructions),
        "tools": [t for t in TOOLS if t["name"] in ("search_web", "send_email")],
        "tool_calls": [
            {"name": "search_web", "arguments": {"query": query, "num_results": n}},
            {"name": "send_email", "arguments": {
                "to": recipient,
                "subject": f"Search results: {query}",
                "body": f"Here are the top {n} results for: {query}",
            }},
        ],
        "category": "multi_step",
        "num_steps": 2,
    }


def make_multistep_weather_reminder(rng: random.Random) -> dict:
    city = rng.choice(CITIES)
    date = rng.choice(DATE_PATTERNS[:10])
    time = rng.choice(TIME_PATTERNS)
    msg = rng.choice(["Check weather before departure", "Pack an umbrella",
                      "Wear a coat", "Bring sunscreen", "Prepare for rain"])
    instructions = [
        f"Check the weather in {city} and set a reminder for '{msg}' on {date} at {time}.",
        f"What's the weather in {city}? Also create a reminder: '{msg}' at {time} on {date}.",
        f"Get {city} weather and remind me to '{msg}' on {date}.",
    ]
    return {
        "instruction": rng.choice(instructions),
        "tools": [t for t in TOOLS if t["name"] in ("get_weather", "set_reminder")],
        "tool_calls": [
            {"name": "get_weather", "arguments": {"city": city, "units": "C"}},
            {"name": "set_reminder", "arguments": {"message": msg, "datetime": f"{date}T{time}:00"}},
        ],
        "category": "multi_step",
        "num_steps": 2,
    }


def make_multistep_calendar_email_reminder(rng: random.Random) -> dict:
    """3-step: create calendar event → send email invite → set reminder."""
    title = rng.choice(["Team sync", "Project review", "Client call", "Quarterly planning",
                        "Design sprint", "1:1 with manager", "Product demo", "Sprint retrospective"])
    date = rng.choice(DATE_PATTERNS[:15])
    time = rng.choice(TIME_PATTERNS)
    attendee = rng.choice(PEOPLE)
    recipient = rng.choice([p for p in PEOPLE if p != attendee])
    msg = rng.choice(REMINDER_MESSAGES)
    instructions = [
        f"Schedule a '{title}' on {date} at {time}, email {recipient} the invite, and set a reminder to prepare beforehand.",
        f"Create a calendar event for '{title}' ({date} {time}), then send {recipient} an email, then remind me about prep.",
        f"I need to: (1) add '{title}' to my calendar for {date} at {time}, (2) email {recipient} about it, (3) set a reminder.",
        f"Book '{title}' on {date} at {time} with {attendee}, email {recipient} the details, and set a '{msg}' reminder.",
    ]
    return {
        "instruction": rng.choice(instructions),
        "tools": [t for t in TOOLS if t["name"] in ("create_calendar_event", "send_email", "set_reminder")],
        "tool_calls": [
            {"name": "create_calendar_event", "arguments": {
                "title": title, "date": date, "time": time, "attendees": [attendee],
            }},
            {"name": "send_email", "arguments": {
                "to": recipient,
                "subject": f"Invite: {title} on {date}",
                "body": f"You are invited to '{title}' on {date} at {time}. Please confirm attendance.",
            }},
            {"name": "set_reminder", "arguments": {
                "message": f"Prepare for {title}", "datetime": f"{date}T08:00:00",
            }},
        ],
        "category": "multi_step",
        "num_steps": 3,
    }


def make_multistep_search_translate_email(rng: random.Random) -> dict:
    """3-step: search → translate result → email it."""
    query = rng.choice(SEARCH_QUERIES)
    target, lang_name = rng.choice(list(LANGUAGES.items()))
    recipient = rng.choice(PEOPLE)
    instructions = [
        f"Search for '{query}', translate the summary to {lang_name}, then email it to {recipient}.",
        f"Look up '{query}', convert the result to {lang_name}, and forward to {recipient}.",
        f"Find info on '{query}', translate it into {lang_name}, then send to {recipient}.",
    ]
    return {
        "instruction": rng.choice(instructions),
        "tools": [t for t in TOOLS if t["name"] in ("search_web", "translate_text", "send_email")],
        "tool_calls": [
            {"name": "search_web", "arguments": {"query": query, "num_results": 5}},
            {"name": "translate_text", "arguments": {"text": f"Search results for: {query}", "target_lang": target}},
            {"name": "send_email", "arguments": {
                "to": recipient,
                "subject": f"[{lang_name}] {query}",
                "body": f"Translated search results for '{query}' in {lang_name}.",
            }},
        ],
        "category": "multi_step",
        "num_steps": 3,
    }


def make_multistep_stock_exchange_calculate(rng: random.Random) -> dict:
    """3-step: stock price → exchange rate → calculate converted value."""
    ticker = rng.choice(TICKERS)
    from_c = "USD"
    to_c = rng.choice([c for c in CURRENCIES[:10] if c != "USD"])
    shares = rng.randint(1, 100)
    instructions = [
        f"Get {ticker} stock price, get the {from_c}/{to_c} exchange rate, then calculate the value of {shares} shares in {to_c}.",
        f"What is {shares} shares of {ticker} worth in {to_c}? Get the stock price, exchange rate, then compute.",
        f"Fetch {ticker} price and the {from_c} to {to_c} rate, then calculate total for {shares} shares.",
    ]
    return {
        "instruction": rng.choice(instructions),
        "tools": [t for t in TOOLS if t["name"] in ("get_stock_price", "get_exchange_rate", "calculate")],
        "tool_calls": [
            {"name": "get_stock_price", "arguments": {"symbol": ticker, "currency": from_c}},
            {"name": "get_exchange_rate", "arguments": {"from_currency": from_c, "to_currency": to_c}},
            {"name": "calculate", "arguments": {"expression": f"{shares} * price * rate"}},
        ],
        "category": "multi_step",
        "num_steps": 3,
    }


def make_ambiguous_weather_vs_news(rng: random.Random) -> dict:
    """Ambiguous: 'what's happening in X' — weather, not news."""
    city = rng.choice(CITIES)
    instructions = [
        f"What's happening with the weather in {city}?",
        f"What are the current conditions in {city} outside?",
        f"Tell me the latest outdoor conditions in {city}.",
        f"How's {city} looking weather-wise today?",
    ]
    return {
        "instruction": rng.choice(instructions),
        "tools": [t for t in TOOLS if t["name"] in ("get_weather", "get_news")],
        "tool_calls": [{"name": "get_weather", "arguments": {"city": city, "units": "C"}}],
        "category": "weather",
        "num_steps": 1,
    }


def make_ambiguous_calculate_vs_convert(rng: random.Random) -> dict:
    """Ambiguous: unit conversion phrased as a calculation."""
    from_unit, to_unit, val_fn = rng.choice(UNIT_CONVERSIONS)
    value = val_fn(rng)
    instructions = [
        f"Calculate how much {value} {from_unit} is in {to_unit}.",
        f"Compute the {to_unit} equivalent of {value} {from_unit}.",
        f"Figure out {value} {from_unit} expressed in {to_unit}.",
        f"Work out: {value} {from_unit} in {to_unit}.",
    ]
    return {
        "instruction": rng.choice(instructions),
        "tools": [t for t in TOOLS if t["name"] in ("convert_units", "calculate")],
        "tool_calls": [{"name": "convert_units", "arguments": {
            "value": value, "from_unit": from_unit, "to_unit": to_unit,
        }}],
        "category": "conversion",
        "num_steps": 1,
    }


def make_ambiguous_search_vs_news(rng: random.Random) -> dict:
    """Ambiguous: 'find latest info on X' — web search, not news."""
    topic = rng.choice(NEWS_TOPICS)
    instructions = [
        f"Find the latest information about {topic} online.",
        f"Search for recent developments in {topic}.",
        f"Look up what's new with {topic} on the web.",
        f"Get me web results about {topic}.",
    ]
    return {
        "instruction": rng.choice(instructions),
        "tools": [t for t in TOOLS if t["name"] in ("search_web", "get_news")],
        "tool_calls": [{"name": "search_web", "arguments": {"query": topic, "num_results": 5}}],
        "category": "search",
        "num_steps": 1,
    }


# ── Main generator ─────────────────────────────────────────────────────────────

SINGLE_GENERATORS_WEIGHTS = [
    (make_weather_example, 4),
    (make_stock_example, 3),
    (make_translate_example, 3),
    (make_calculate_example, 4),
    (make_convert_units_example, 3),
    (make_reminder_example, 3),
    (make_exchange_rate_example, 2),
    (make_news_example, 3),
    (make_search_example, 3),
    (make_music_example, 2),
    (make_directions_example, 3),
    (make_send_email_example, 3),
    # Ambiguous examples — model must discriminate between similar tools
    (make_ambiguous_weather_vs_news, 2),
    (make_ambiguous_calculate_vs_convert, 2),
    (make_ambiguous_search_vs_news, 2),
]

MULTI_GENERATORS_WEIGHTS = [
    (make_multistep_weather_stock, 3),
    (make_multistep_news_email, 2),
    (make_multistep_directions_hotel, 2),
    (make_multistep_exchange_convert, 2),
    (make_multistep_search_email, 2),
    (make_multistep_weather_reminder, 2),
    # 3-step chains
    (make_multistep_calendar_email_reminder, 3),
    (make_multistep_search_translate_email, 2),
    (make_multistep_stock_exchange_calculate, 2),
]

# 65% single, 35% multi-step (up from 30% to give model more chain exposure)
ALL_GENERATORS = (
    [(g, w * 65) for g, w in SINGLE_GENERATORS_WEIGHTS] +
    [(g, w * 35) for g, w in MULTI_GENERATORS_WEIGHTS]
)


def _add_distractor_tools(example: dict, rng: random.Random) -> None:
    """Inject 3-5 distractor tools so the model must select from a candidate set."""
    needed = {t["name"] for t in example.get("tools", [])}
    distractors = [t for t in TOOLS if t["name"] not in needed]
    n = rng.randint(3, min(5, len(distractors)))
    extra = rng.sample(distractors, n)
    # Shuffle so the correct tool isn't always first
    all_tools = example["tools"] + extra
    rng.shuffle(all_tools)
    example["tools"] = all_tools


def generate_dataset(num_samples: int, seed: int) -> list:
    rng = random.Random(seed)
    generators = [g for g, _ in ALL_GENERATORS]
    weights = [w for _, w in ALL_GENERATORS]
    examples = []
    for _ in range(num_samples):
        gen_fn = rng.choices(generators, weights=weights, k=1)[0]
        example = gen_fn(rng)

        # Add distractor tools to make tool selection non-trivial
        _add_distractor_tools(example, rng)

        # Build text: serialize ALL tool calls (fixes multi-step training)
        call_blocks = "\n".join(
            f"<tool_call>\n{json.dumps(tc)}\n</tool_call>"
            for tc in example["tool_calls"]
        )
        example["text"] = (
            f"[{example['category']}] "
            f"USER: {example['instruction']}\n"
            f"ASSISTANT: {call_blocks}"
        )
        examples.append(example)
    return examples


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic tool-use data")
    parser.add_argument("--num-samples", type=int, default=15000,
                        help="Number of examples to generate (default: 15000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./data/raw/synthetic")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.num_samples} synthetic examples (seed={args.seed})...")
    examples = generate_dataset(args.num_samples, args.seed)

    # Split into single-step and multi-step files for clarity
    single = [e for e in examples if e["num_steps"] == 1]
    multi  = [e for e in examples if e["num_steps"] > 1]

    single_path = out_dir / "synthetic_single.jsonl"
    multi_path  = out_dir / "synthetic_multistep.jsonl"

    with open(single_path, "w") as f:
        for e in single:
            f.write(json.dumps(e) + "\n")

    with open(multi_path, "w") as f:
        for e in multi:
            f.write(json.dumps(e) + "\n")

    total = len(single) + len(multi)
    print(f"✅ Saved {len(single):,} single-step examples → {single_path}")
    print(f"✅ Saved {len(multi):,}  multi-step examples  → {multi_path}")
    print(f"✅ Total: {total:,} examples")


if __name__ == "__main__":
    main()
