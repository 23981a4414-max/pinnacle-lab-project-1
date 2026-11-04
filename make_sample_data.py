"""Generate a small SYNTHETIC demo dataset so the pipeline runs end to end
without internet access. Replace it with a real dataset (see README) for
meaningful results."""
import random
import pandas as pd

random.seed(42)

REAL_SUBJ = ["The central bank", "Health officials", "The city council", "Researchers at the university",
             "The finance ministry", "Election commission officials", "The transport department",
             "Scientists", "The national statistics office", "The school board"]
REAL_VERB = ["announced", "reported", "confirmed", "published", "approved", "released", "reviewed"]
REAL_OBJ = ["a quarterly report on economic growth", "new guidelines after a public consultation",
            "the annual budget following committee review", "findings from a peer-reviewed study",
            "updated figures on regional employment", "a plan to upgrade public transport",
            "results of the latest vaccination survey", "a statement citing official data"]

FAKE_SUBJ = ["SHOCKING:", "BREAKING!!!", "You won't believe:", "Secret memo LEAKED:", "WAKE UP:",
             "Doctors HATE this:", "EXPOSED:", "They don't want you to know:"]
FAKE_BODY = ["celebrity secretly replaced by a clone, insiders say", "miracle cure hidden by big pharma for decades",
             "government hiding alien base, anonymous source claims", "one weird trick makes millions overnight",
             "politician caught in massive cover-up, share before it is deleted",
             "5G towers behind mysterious illness, 100% proof", "banks planning to freeze all accounts tomorrow",
             "famous actor dead hoax spreads, click to see the TRUTH"]


REGIONS = ["in Hyderabad", "in Vijayawada", "in Delhi", "in Mumbai", "in Chennai", "in Kolkata", "in Pune",
           "in Bengaluru", "in Guntur", "in Visakhapatnam", "this week", "on Monday", "on Friday", "this month"]
FILLER = ["according to local reports", "as the debate continues", "amid growing public interest",
          "say sources", "as people react online", "ahead of the weekend", "following days of discussion", ""]


def real():
    return f"{random.choice(REAL_SUBJ)} {random.choice(REAL_VERB)} {random.choice(REAL_OBJ)} {random.choice(REGIONS)} {random.choice(FILLER)}.".replace("  ", " ")


def fake():
    return f"{random.choice(FAKE_SUBJ)} {random.choice(FAKE_BODY)} {random.choice(REGIONS)} {random.choice(FILLER)}!".replace("  ", " ")


rows = [(real(), 1) for _ in range(1500)] + [(fake(), 0) for _ in range(1500)]
for i in range(0, len(rows), 25):
    t, l = rows[i]
    rows[i] = (t + " " + (fake() if l == 1 else real()), l)
random.shuffle(rows)
pd.DataFrame(rows, columns=["text", "label"]).to_csv("data/sample_news.csv", index=False)
print("Wrote data/sample_news.csv with", len(rows), "rows (label 1=REAL, 0=FAKE)")
