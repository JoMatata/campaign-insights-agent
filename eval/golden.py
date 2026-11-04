"""Golden questions. Expected answers come from running the reference SQL, so they
always match the generated database.

Each SQL returns one row. Numeric columns are the expected numbers; text columns must
appear in the answer. `percent` marks rates the model may report as 4.7 or 0.047.
`accept` lists alternative spellings for text answers.
"""

CHANNEL = "FROM daily_performance d JOIN campaigns c USING (campaign_id)"

GOLDEN = [
    {"id": "q01", "question": "What was the total ad spend across all campaigns, in KES?",
     "sql": "SELECT sum(spend) FROM daily_performance"},
    {"id": "q02", "question": "How much revenue did the campaigns generate in total?",
     "sql": "SELECT sum(revenue) FROM daily_performance"},
    {"id": "q03", "question": "What is the overall ROAS across all campaigns?",
     "sql": "SELECT sum(revenue) / sum(spend) FROM daily_performance"},
    {"id": "q04", "question": "Which channel has the highest ROAS?",
     "sql": f"SELECT c.channel {CHANNEL} GROUP BY 1 ORDER BY sum(revenue)/sum(spend) DESC LIMIT 1",
     "accept": ["google search", "search"]},
    {"id": "q05", "question": "What is the ROAS of the google_search channel?",
     "sql": f"SELECT sum(revenue)/sum(spend) {CHANNEL} WHERE c.channel = 'google_search'"},
    {"id": "q06", "question": "What is the click-through rate of TikTok campaigns overall?",
     "sql": f"SELECT sum(clicks)/sum(impressions) {CHANNEL} WHERE c.channel = 'tiktok'", "percent": True},
    {"id": "q07", "question": "What are the conversion rates (conversions per click) for TikTok and Meta?",
     "sql": f"SELECT sum(conversions) FILTER (WHERE channel='tiktok') / sum(clicks) FILTER (WHERE channel='tiktok'), "
            f"sum(conversions) FILTER (WHERE channel='meta') / sum(clicks) FILTER (WHERE channel='meta') {CHANNEL}",
     "percent": True},
    {"id": "q08", "question": "What was the cost per acquisition (CPA) for campaign C01?",
     "sql": "SELECT sum(spend)/sum(conversions) FROM daily_performance WHERE campaign_id = 'C01'"},
    {"id": "q09", "question": "How many conversions came from Nairobi county?",
     "sql": "SELECT sum(conversions) FROM daily_performance WHERE county = 'Nairobi'"},
    {"id": "q10", "question": "What share of all clicks came from mobile devices?",
     "sql": "SELECT sum(clicks) FILTER (WHERE device='mobile') / sum(clicks) FROM daily_performance",
     "percent": True},
    {"id": "q11", "question": "Compare the conversion rate on desktop vs mobile.",
     "sql": "SELECT sum(conversions) FILTER (WHERE device='desktop') / sum(clicks) FILTER (WHERE device='desktop'), "
            "sum(conversions) FILTER (WHERE device='mobile') / sum(clicks) FILTER (WHERE device='mobile') "
            "FROM daily_performance", "percent": True},
    {"id": "q12", "question": "Which county had the highest ROAS?",
     "sql": "SELECT county FROM daily_performance GROUP BY 1 ORDER BY sum(revenue)/sum(spend) DESC LIMIT 1"},
    {"id": "q13", "question": "How many conversions were there on 28 May 2026?",
     "sql": "SELECT sum(conversions) FROM daily_performance WHERE date = DATE '2026-05-28'"},
    {"id": "q14", "question": "Conversions dropped around the end of May. What was the average daily "
                              "conversions in May 2026 excluding 28 and 29 May, and what was it on 28 May?",
     "sql": "WITH d AS (SELECT date, sum(conversions) conv FROM daily_performance "
            "WHERE date BETWEEN '2026-05-01' AND '2026-05-31' GROUP BY 1) "
            "SELECT avg(conv) FILTER (WHERE date NOT IN ('2026-05-28','2026-05-29')), "
            "sum(conv) FILTER (WHERE date = '2026-05-28') FROM d"},
    {"id": "q15", "question": "Among campaigns with the 'conversions' objective, which has the highest CPA?",
     "sql": f"SELECT c.campaign_id, c.campaign_name {CHANNEL} WHERE c.objective = 'conversions' "
            "GROUP BY 1, 2 ORDER BY sum(spend)/sum(conversions) DESC LIMIT 1", "any_text": True},
    {"id": "q16", "question": "What percentage of its planned budget did campaign C05 spend?",
     "sql": "SELECT sum(d.spend) / max(c.budget_kes) FROM daily_performance d JOIN campaigns c "
            "USING (campaign_id) WHERE campaign_id = 'C05'", "percent": True},
    {"id": "q17", "question": "How many campaigns ran on TikTok?",
     "sql": "SELECT count(*) FROM campaigns WHERE channel = 'tiktok'"},
    {"id": "q18", "question": "How many video creatives are there across all campaigns?",
     "sql": "SELECT count(*) FROM creatives WHERE format = 'video'"},
    {"id": "q19", "question": "How much was spent on ads in March 2026?",
     "sql": "SELECT sum(spend) FROM daily_performance WHERE date BETWEEN '2026-03-01' AND '2026-03-31'"},
    {"id": "q20", "question": "Which month in 2026 had the highest revenue?",
     "sql": "SELECT strftime(date, '%B') FROM daily_performance GROUP BY 1 ORDER BY sum(revenue) DESC LIMIT 1"},
    {"id": "q21", "question": "How many feedback snippets mention M-Pesa (spelled M-Pesa or mpesa)?",
     "sql": "SELECT count(*) FROM feedback WHERE text ILIKE '%m-pesa%' OR text ILIKE '%mpesa%'"},
    {"id": "q22", "question": "What is the average rating of app reviews?",
     "sql": "SELECT avg(rating) FROM feedback WHERE source = 'app_review'"},
    {"id": "q23", "question": "How many ad comments were left on campaign C11?",
     "sql": "SELECT count(*) FROM feedback WHERE source = 'ad_comment' AND campaign_id = 'C11'"},
    {"id": "q24", "question": "Compare the conversion rates of C10 (Meta Madaraka Day) and C11 (TikTok Madaraka Day).",
     "sql": "SELECT sum(conversions) FILTER (WHERE campaign_id='C10') / sum(clicks) FILTER (WHERE campaign_id='C10'), "
            "sum(conversions) FILTER (WHERE campaign_id='C11') / sum(clicks) FILTER (WHERE campaign_id='C11') "
            "FROM daily_performance", "percent": True},
    {"id": "q25", "question": "What were the CTRs of C01 and C02, the two always-on search campaigns?",
     "sql": "SELECT sum(clicks) FILTER (WHERE campaign_id='C01') / sum(impressions) FILTER (WHERE campaign_id='C01'), "
            "sum(clicks) FILTER (WHERE campaign_id='C02') / sum(impressions) FILTER (WHERE campaign_id='C02') "
            "FROM daily_performance", "percent": True},
]
