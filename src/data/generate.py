"""Generate the synthetic Tausi Foods campaign database.

Tausi Foods is a made-up Kenyan consumer brand (cooking oil, flour, snacks,
juice) that sells online with home delivery. Nothing here is real data.

Planted patterns, so the agent has something to find:
- Search converts best; TikTok gets cheap clicks that convert poorly.
- Desktop converts better than mobile, but most traffic is mobile.
- An M-Pesa STK push outage on 28-29 May cuts conversions and spikes payment complaints.
- Mombasa and Kisumu get more delivery complaints.

Run: python -m src.data.generate
"""
import argparse
from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd

from src import config

SEED = 42
COUNTIES = {"Nairobi": 0.38, "Mombasa": 0.13, "Kiambu": 0.12, "Nakuru": 0.09,
            "Kisumu": 0.08, "Uasin Gishu": 0.07, "Machakos": 0.07, "Kakamega": 0.06}
DEVICES = {"mobile": (0.78, 0.85), "desktop": (0.17, 1.45), "tablet": (0.05, 1.0)}  # share, CVR multiplier
OUTAGE_DAYS = {date(2026, 5, 28), date(2026, 5, 29)}

# channel: (ctr, cvr, cpm in KES, average order value in KES)
CHANNELS = {
    "google_search": (0.045, 0.060, 900, 2300),
    "meta": (0.012, 0.028, 260, 1900),
    "tiktok": (0.019, 0.009, 180, 1500),
    "youtube": (0.006, 0.015, 220, 1800),
    "display": (0.004, 0.022, 120, 2000),
    "x": (0.008, 0.010, 200, 1700),
}

# id, name, channel, objective, start, end, budget (KES)
CAMPAIGNS = [
    ("C01", "Search Brand Always-On", "google_search", "conversions", "2026-01-01", "2026-06-30", 2_400_000),
    ("C02", "Search Generic Cooking Oil", "google_search", "conversions", "2026-01-15", "2026-06-30", 1_500_000),
    ("C03", "Meta Back to School", "meta", "conversions", "2026-01-05", "2026-02-15", 900_000),
    ("C04", "Meta Valentine Hampers", "meta", "conversions", "2026-02-01", "2026-02-16", 450_000),
    ("C05", "TikTok Sheng Challenge", "tiktok", "awareness", "2026-03-01", "2026-04-15", 800_000),
    ("C06", "TikTok Easter Recipes", "tiktok", "traffic", "2026-03-20", "2026-04-10", 400_000),
    ("C07", "YouTube Recipe Series", "youtube", "awareness", "2026-02-01", "2026-05-31", 1_100_000),
    ("C08", "Display Retargeting", "display", "conversions", "2026-01-01", "2026-06-30", 700_000),
    ("C09", "Meta Ramadan Iftar Packs", "meta", "conversions", "2026-02-18", "2026-03-20", 650_000),
    ("C10", "Meta Madaraka Day Deals", "meta", "conversions", "2026-05-20", "2026-06-05", 550_000),
    ("C11", "TikTok Madaraka Day", "tiktok", "conversions", "2026-05-20", "2026-06-05", 500_000),
    ("C12", "Search Madaraka Day", "google_search", "conversions", "2026-05-20", "2026-06-05", 600_000),
    ("C13", "X Launch Teasers", "x", "awareness", "2026-01-10", "2026-02-10", 250_000),
    ("C14", "YouTube Shorts Snacks", "youtube", "traffic", "2026-04-01", "2026-06-30", 600_000),
    ("C15", "Display Prospecting", "display", "awareness", "2026-03-01", "2026-06-30", 500_000),
    ("C16", "Meta Wholesale Leads", "meta", "traffic", "2026-04-01", "2026-05-15", 350_000),
    ("C17", "Search Competitor Terms", "google_search", "conversions", "2026-04-01", "2026-06-30", 800_000),
    ("C18", "TikTok Always-On Snacks", "tiktok", "traffic", "2026-01-01", "2026-06-30", 1_200_000),
]


def _dates(start, end):
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    return [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]


def make_performance(rng):
    rows = []
    for cid, _, channel, _, start, end, budget in CAMPAIGNS:
        ctr, cvr, cpm, aov = CHANNELS[channel]
        # each campaign is a bit better or worse than its channel average
        ctr *= rng.lognormal(0, 0.15)
        cvr *= rng.lognormal(0, 0.15)
        days = _dates(start, end)
        daily_spend = budget * rng.uniform(0.85, 1.02) / len(days)
        for d in days:
            weekday = 1.12 if d.weekday() >= 5 else 0.95
            outage = 0.35 if d in OUTAGE_DAYS else 1.0
            for county, c_share in COUNTIES.items():
                for device, (d_share, d_cvr) in DEVICES.items():
                    spend = daily_spend * c_share * d_share * weekday * rng.lognormal(0, 0.2)
                    imps = int(spend / cpm * 1000)
                    clicks = rng.binomial(imps, min(ctr * rng.lognormal(0, 0.1), 0.5))
                    convs = rng.binomial(clicks, min(cvr * d_cvr * outage, 0.9))
                    revenue = convs * aov * rng.normal(1, 0.08) if convs else 0.0
                    rows.append((cid, d, county, device, imps, clicks, convs,
                                 round(spend, 2), round(max(revenue, 0), 2)))
    return pd.DataFrame(rows, columns=["campaign_id", "date", "county", "device", "impressions",
                                       "clicks", "conversions", "spend", "revenue"])


def make_creatives(rng):
    formats = {"google_search": ["search_text"], "display": ["image", "carousel"],
               "meta": ["image", "video", "carousel"], "tiktok": ["video"],
               "youtube": ["video"], "x": ["image", "video"]}
    rows = []
    for cid, name, channel, *_ in CAMPAIGNS:
        for i in range(rng.integers(2, 5)):
            rows.append((f"{cid}-CR{i + 1}", cid, str(rng.choice(formats[channel])),
                         str(rng.choice(["english", "swahili", "sheng"], p=[0.5, 0.3, 0.2])),
                         f"{name} - variant {i + 1}"))
    return pd.DataFrame(rows, columns=["creative_id", "campaign_id", "format", "language", "headline"])


# ---------- feedback text ----------

PRODUCTS = ["cooking oil", "the 5L oil jerrycan", "maize flour", "unga", "Tausi crisps",
            "mango juice", "the snack box", "tea leaves", "chapati flour", "the iftar pack"]
TOWNS = ["Kilimani", "Rongai", "Nyali", "Thika", "Westlands", "Kitengela", "Ruaka",
         "Bamburi", "Milimani", "Juja", "Syokimau", "Eldoret town"]

THEMES = {
    "delivery": [
        "my order took {days} days to arrive in {town}",
        "delivery was supposed to be same day but it came after {days} days",
        "the rider kept calling for directions then disappeared",
        "still waiting for my {product}, ordered last week",
        "order imechelewa sana, {days} days na bado",
        "rider alipotea, hakuna mtu ameleta {product}",
        "why does delivery to {town} take forever",
        "they marked it delivered but nothing came",
        "mzigo haujafika hadi sasa, nimengoja {days} days",
        "delivery window was 2pm to 5pm, rider showed up at 9pm",
        "the tracking link never updates",
        "nimengoja rider tangu asubuhi",
    ],
    "pricing": [
        "{product} is now KES {price}, that is too expensive",
        "bei iko juu sana compared to the supermarket",
        "prices went up again this month",
        "the offer price on the ad was different at checkout",
        "delivery fee of KES {fee} is more than the item",
        "ghali sana mazee, {product} kwa {price}",
        "why is the online price higher than in the shop",
        "the discount code did not apply",
        "bei imepanda bila notice",
        "KES {price} for {product}? no thanks",
        "the promo said 20% off but I paid full price",
    ],
    "payment": [
        "M-Pesa STK push never came through",
        "money was deducted from M-Pesa but the order shows unpaid",
        "pesa imekatwa lakini order haijaconfirm",
        "tried to pay with M-Pesa three times and it failed",
        "STK push haikuja, nimejaribu mara {days}",
        "I paid via the till number and I am still waiting for the refund",
        "checkout keeps timing out at the M-Pesa step",
        "nililipa double na hakuna refund",
        "payment failed but M-Pesa sent a confirmation message",
        "card payment works but mpesa is always failing",
        "nimelipa KES {price} kwa mpesa na bado inasema pending",
    ],
    "quality": [
        "the {product} was expired when it arrived",
        "{product} packaging was leaking",
        "quality imeshuka, {product} si kama zamani",
        "the crisps were stale and broken",
        "juice ilikuwa na taste mbaya",
        "found the seal on the {product} already broken",
        "the flour had weevils inside",
        "{product} tastes different from before, not good",
        "oil ilikuwa imeganda na rangi si poa",
        "received a damaged {product}, the box was crushed",
    ],
    "ad_annoyance": [
        "I keep seeing this ad every time I open TikTok",
        "hii ad inaboo, nimeiona mara mia",
        "stop showing me the same video ad",
        "too many ads from Tausi on my feed",
        "the ad plays before every YouTube video, so annoying",
        "nimechoka na hii ad kila saa",
        "this ad is loud and it autoplays",
        "skip button never shows on this ad",
        "why is this ad following me everywhere",
        "manze hii ad imezidi",
    ],
    "service": [
        "customer care never picks up the phone",
        "I sent three emails and got no reply",
        "customer care hawashiki simu",
        "the agent on chat was rude",
        "waited {days} hours on the chat for someone to respond",
        "support closed my ticket without solving anything",
        "nimetuma message WhatsApp hakuna jibu",
        "they promised a callback and never called",
        "the support team keeps asking for the same order number",
        "customer service iko slow sana",
    ],
    "positive": [
        "the {product} arrived quickly and well packed",
        "great quality, my family loves the {product}",
        "poa sana, nitaorder tena",
        "{product} iko fiti, value for money",
        "delivery to {town} was faster than I expected",
        "best chapati flour I have used",
        "nice recipes in the YouTube series",
        "M-Pesa payment was smooth this time",
        "the iftar pack was worth it",
        "service ni poa, rider was polite",
        "love the new snack flavours",
    ],
}
THEME_SHARE = {"delivery": 0.20, "pricing": 0.15, "payment": 0.13, "quality": 0.14,
               "ad_annoyance": 0.12, "service": 0.10, "positive": 0.16}
OPENERS = ["", "", "", "honestly ", "Tausi, ", "bana ", "hi, ", "so ", "wueh ", "again, "]
CLOSERS = ["", "", "", ". please sort this out", ". sio poa", ". will not order again",
           ". fix this", " pls", ". very disappointed", " kwa kweli"]
POSITIVE_CLOSERS = ["", "", ". asante", ". 10/10", ". keep it up", " kwa kweli"]
SOURCES = {"ad_comment": 0.35, "app_review": 0.25, "support_chat": 0.25, "survey": 0.15}


def _fill(template, rng):
    return template.format(product=rng.choice(PRODUCTS), town=rng.choice(TOWNS),
                           days=rng.integers(2, 8), price=rng.integers(3, 30) * 50,
                           fee=rng.integers(2, 6) * 50)


def make_feedback(rng, n=3000):
    themes = list(THEME_SHARE)
    probs = np.array(list(THEME_SHARE.values()))
    dates = _dates("2026-01-01", "2026-06-30")
    campaign_ids = [c[0] for c in CAMPAIGNS]
    rows, seen = [], set()
    while len(rows) < n:
        d = dates[rng.integers(len(dates))]
        county = str(rng.choice(list(COUNTIES), p=list(COUNTIES.values())))
        p = probs.copy()
        if d in OUTAGE_DAYS or (d - date(2026, 5, 28)).days in (-1, 2):
            p[themes.index("payment")] *= 6
        if county in ("Mombasa", "Kisumu"):
            p[themes.index("delivery")] *= 2
        theme = str(rng.choice(themes, p=p / p.sum()))
        text = rng.choice(OPENERS) + _fill(rng.choice(THEMES[theme]), rng)
        # about 12% mention a second theme, which makes classification harder
        if rng.random() < 0.12:
            other = str(rng.choice([t for t in themes if t != theme]))
            text += rng.choice([", also ", ". and ", ". plus "]) + _fill(rng.choice(THEMES[other]), rng)
        text += rng.choice(POSITIVE_CLOSERS if theme == "positive" else CLOSERS)
        text = text.lower() if rng.random() < 0.4 else text[0].upper() + text[1:]
        if text in seen:
            continue
        seen.add(text)
        source = str(rng.choice(list(SOURCES), p=list(SOURCES.values())))
        if theme == "ad_annoyance" and rng.random() < 0.7:
            source = "ad_comment"
        campaign = str(rng.choice(campaign_ids)) if source == "ad_comment" else None
        if source in ("app_review", "survey"):
            rating = int(rng.integers(4, 6)) if theme == "positive" else int(rng.integers(1, 4))
        else:
            rating = None
        rows.append((len(rows) + 1, d, source, campaign, county, rating, text, theme))
    df = pd.DataFrame(rows, columns=["feedback_id", "created_at", "source", "campaign_id",
                                     "county", "rating", "text", "theme"])
    df["rating"] = df["rating"].astype("Int64")
    return df


def build(db_path=config.DB_PATH, seed=SEED, n_feedback=3000):
    rng = np.random.default_rng(seed)
    campaigns = pd.DataFrame(CAMPAIGNS, columns=["campaign_id", "campaign_name", "channel",
                                                 "objective", "start_date", "end_date", "budget_kes"])
    campaigns["start_date"] = pd.to_datetime(campaigns["start_date"]).dt.date
    campaigns["end_date"] = pd.to_datetime(campaigns["end_date"]).dt.date
    perf = make_performance(rng)
    creatives = make_creatives(rng)
    feedback = make_feedback(rng, n_feedback)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    con = duckdb.connect(str(db_path))
    for name, df in [("campaigns", campaigns), ("daily_performance", perf),
                     ("creatives", creatives), ("feedback", feedback.drop(columns="theme"))]:
        con.register("tmp_df", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM tmp_df")
        con.unregister("tmp_df")
    con.close()

    # theme labels stay outside the database so the agent cannot read the answer key
    feedback[["feedback_id", "text", "theme"]].to_csv(db_path.parent / "feedback_truth.csv", index=False)
    labelled = feedback.groupby("theme").sample(frac=300 / len(feedback), random_state=seed)
    labelled[["feedback_id", "text", "theme"]].sort_values("feedback_id").to_csv(
        db_path.parent / "labelled_feedback.csv", index=False)
    return {"daily_performance": len(perf), "feedback": len(feedback), "labelled": len(labelled)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    print(build(seed=args.seed))
