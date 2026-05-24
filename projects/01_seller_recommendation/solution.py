"""
Use Case: Hyper-Personalised Seller Recommendation & Messaging System
-----------------------------------------------------------------------
Problem : Generate personalised program recommendations and email content
          for millions of sellers using behavioural signals and LLM reasoning.
Approach: Behavioural signal extraction → Seller context builder →
          LLM recommendation reasoning → Personalised email generation →
          Subject line optimisation → Sequential engagement loop.
"""

import json
from dataclasses import dataclass, field
from typing import Optional
from openai import OpenAI

client = OpenAI()


@dataclass
class SellerProfile:
    seller_id: str
    categories: list[str]
    monthly_revenue: float
    fulfillment_method: str
    sponsored_ads_active: bool
    deal_participation: bool
    enhanced_content_enabled: bool
    seller_central_logins_30d: int
    avg_listing_update_days: float
    tenure_months: int
    adopted_programs: list[str] = field(default_factory=list)
    last_engagement_response: Optional[str] = None


@dataclass
class GrowthProgram:
    name: str
    benefit: str
    eligibility_hint: str


PROGRAMS = [
    GrowthProgram("Fulfillment Program",   "Faster delivery, higher buy-box win rate",       "Self-ship sellers with >50 orders/month"),
    GrowthProgram("Advertising Program",   "Increase product visibility and organic rank",    "Sellers with >10 active listings"),
    GrowthProgram("Deals & Promotions",    "Drive volume spikes, clear slow-moving inventory","Sellers with excess or seasonal inventory"),
    GrowthProgram("Enhanced Content",      "Improve conversion with rich product pages",      "Sellers with <50% enhanced content coverage"),
    GrowthProgram("Warehousing Program",   "Reduce storage costs, improve fulfilment speed",  "High-volume sellers shipping >200 units/month"),
]


class BehaviouralSignalExtractor:
    def extract(self, seller: SellerProfile) -> dict:
        return {
            "high_login_activity":      seller.seller_central_logins_30d > 15,
            "frequent_listing_updates": seller.avg_listing_update_days < 7,
            "no_ads":                   not seller.sponsored_ads_active,
            "no_deals":                 not seller.deal_participation,
            "no_enhanced_content":      not seller.enhanced_content_enabled,
            "self_ship":                seller.fulfillment_method == "self-ship",
            "high_revenue":             seller.monthly_revenue > 10000,
            "long_tenure":              seller.tenure_months > 12,
            "previously_ignored":       seller.last_engagement_response == "ignored",
        }


class LLMRecommendationReasoner:
    def recommend(self, seller: SellerProfile, signals: dict) -> GrowthProgram:
        available = [p for p in PROGRAMS if p.name not in seller.adopted_programs]
        program_list = "\n".join([f"- {p.name}: {p.benefit} ({p.eligibility_hint})" for p in available])
        prompt = f"""You are a seller growth advisor. Recommend ONE program with the highest impact.

Seller Profile:
- Categories: {', '.join(seller.categories)}
- Monthly Revenue: ${seller.monthly_revenue:,.0f}
- Fulfillment: {seller.fulfillment_method}
- Tenure: {seller.tenure_months} months
- Already adopted: {', '.join(seller.adopted_programs) or 'None'}

Active Signals: {json.dumps({k: v for k, v in signals.items() if v}, indent=2)}

Available Programs:
{program_list}

Respond with ONLY the exact program name."""

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        name = resp.choices[0].message.content.strip()
        return next((p for p in PROGRAMS if p.name == name), PROGRAMS[0])


class PersonalisedEmailGenerator:
    def generate(self, seller: SellerProfile, program: GrowthProgram) -> dict:
        prompt = f"""Write a personalised seller growth email recommending the {program.name}.

Seller: {', '.join(seller.categories)} | ${seller.monthly_revenue:,.0f}/month | {seller.tenure_months} months tenure
Benefit: {program.benefit}

Write:
SUBJECT: <compelling subject line, max 60 chars>
BODY:
<3-paragraph personalised email, specific and action-oriented>"""

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        raw   = resp.choices[0].message.content.strip().split("\n")
        return {
            "subject": raw[0].replace("SUBJECT:", "").strip(),
            "body":    "\n".join(raw[2:]).replace("BODY:", "").strip(),
        }


class SellerEngagementOrchestrator:
    def __init__(self):
        self.extractor = BehaviouralSignalExtractor()
        self.reasoner  = LLMRecommendationReasoner()
        self.generator = PersonalisedEmailGenerator()

    def run_cycle(self, seller: SellerProfile) -> dict:
        signals = self.extractor.extract(seller)
        program = self.reasoner.recommend(seller, signals)
        email   = self.generator.generate(seller, program)
        return {"seller_id": seller.seller_id, "recommended": program.name,
                "active_signals": [k for k, v in signals.items() if v], "email": email}


if __name__ == "__main__":
    seller = SellerProfile(
        seller_id="S-100234", categories=["Electronics", "Mobile Accessories"],
        monthly_revenue=18500, fulfillment_method="self-ship",
        sponsored_ads_active=False, deal_participation=False,
        enhanced_content_enabled=False, seller_central_logins_30d=22,
        avg_listing_update_days=4, tenure_months=18, adopted_programs=[],
    )
    result = SellerEngagementOrchestrator().run_cycle(seller)
    print(f"Seller: {result['seller_id']}")
    print(f"Recommended: {result['recommended']}")
    print(f"Active Signals: {result['active_signals']}")
    print(f"Subject: {result['email']['subject']}")
    print(f"Body:\n{result['email']['body']}")
