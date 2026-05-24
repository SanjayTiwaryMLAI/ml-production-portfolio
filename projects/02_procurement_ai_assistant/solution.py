"""
Use Case: Government Procurement AI Assistant — Multi-Agent Workflow
---------------------------------------------------------------------
Problem : Transform a static FAQ chatbot into an intelligent assistant
          capable of answering policy questions AND executing operational tasks.
Approach: Intent detection → Orchestrator → Specialised agents
          (RAG, Database, Ticket, Escalation) → Unified response.
"""

import json
import random
import string
from enum import Enum
from openai import OpenAI

client = OpenAI()


class Intent(str, Enum):
    POLICY_QUESTION = "policy_question"
    BID_STATUS      = "bid_status"
    ORDER_DETAILS   = "order_details"
    RAISE_TICKET    = "raise_ticket"
    TRACK_TICKET    = "track_ticket"
    ESCALATE        = "escalate"
    UNKNOWN         = "unknown"


MOCK_BIDS = {
    "BID-2024-001": {"status": "Under Evaluation", "submitted": "2024-11-01", "evaluator": "Committee A"},
    "BID-2024-002": {"status": "Awarded",          "submitted": "2024-10-15", "evaluator": "Committee B"},
    "BID-2024-003": {"status": "Rejected",         "submitted": "2024-10-20", "reason": "Incomplete documentation"},
}

MOCK_ORDERS = {
    "ORD-5001": {"item": "Office Furniture", "qty": 50,  "status": "Delivered",  "delivery_date": "2024-11-10"},
    "ORD-5002": {"item": "Laptops",          "qty": 20,  "status": "In Transit", "eta": "2024-11-25"},
}

POLICY_DOCS = [
    "Vendors must submit EMD of 2% of bid value.",
    "Bids are evaluated on quality (40%), price (40%), and delivery (20%).",
    "All procurement above Rs 10 Lakh requires a 3-member evaluation committee.",
    "Rejected bids can be appealed within 15 working days.",
    "Delivery period must not exceed 90 days from purchase order date.",
]


class IntentClassifier:
    def classify(self, query: str) -> tuple[Intent, dict]:
        prompt = f"""Classify this procurement query. Respond as JSON.

Query: "{query}"

Intents: policy_question, bid_status, order_details, raise_ticket, track_ticket, escalate, unknown

JSON: {{"intent": "<intent>", "entities": {{"bid_id": null, "order_id": null, "ticket_id": null}}}}"""

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        return Intent(data.get("intent", "unknown")), data.get("entities", {})


class PolicyRAGAgent:
    def answer(self, query: str) -> str:
        relevant = [d for d in POLICY_DOCS if any(w in d.lower() for w in query.lower().split())] or POLICY_DOCS[:2]
        context  = "\n".join(f"- {d}" for d in relevant)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Answer using policy context.\n\nContext:\n{context}\n\nQ: {query}"}],
            temperature=0,
        )
        return resp.choices[0].message.content.strip()


class BidStatusAgent:
    def get_status(self, bid_id: str) -> str:
        bid = MOCK_BIDS.get(bid_id.upper())
        if not bid:
            return f"No bid found with ID {bid_id}."
        return f"Bid {bid_id}:\n" + "\n".join(f"  {k}: {v}" for k, v in bid.items())


class OrderDetailsAgent:
    def get_order(self, order_id: str) -> str:
        order = MOCK_ORDERS.get(order_id.upper())
        if not order:
            return f"No order found with ID {order_id}."
        return f"Order {order_id}:\n" + "\n".join(f"  {k}: {v}" for k, v in order.items())


class TicketAgent:
    def create(self, query: str) -> str:
        tid = "TKT-" + "".join(random.choices(string.digits, k=6))
        return f"Ticket {tid} created.\nIssue: {query[:80]}\nExpected resolution: 2-3 business days."

    def track(self, ticket_id: str) -> str:
        status = random.choice(["Open", "In Progress", "Resolved"])
        return f"Ticket {ticket_id} Status: {status}\nLast updated: 2024-11-15"


class EscalationAgent:
    def escalate(self, query: str) -> str:
        ref = "ESC-" + "".join(random.choices(string.digits, k=5))
        return f"Escalated to senior officer. Ref: {ref}\nA representative will contact you within 4 business hours."


class ProcurementOrchestrator:
    def __init__(self):
        self.classifier = IntentClassifier()
        self.policy     = PolicyRAGAgent()
        self.bid        = BidStatusAgent()
        self.order      = OrderDetailsAgent()
        self.ticket     = TicketAgent()
        self.escalation = EscalationAgent()

    def handle(self, query: str) -> str:
        intent, entities = self.classifier.classify(query)
        print(f"  → Intent: {intent.value} | Entities: {entities}")
        if intent == Intent.POLICY_QUESTION:
            return self.policy.answer(query)
        elif intent == Intent.BID_STATUS:
            bid_id = entities.get("bid_id") or ""
            return self.bid.get_status(bid_id) if bid_id else "Please provide a bid ID."
        elif intent == Intent.ORDER_DETAILS:
            order_id = entities.get("order_id") or ""
            return self.order.get_order(order_id) if order_id else "Please provide an order ID."
        elif intent == Intent.RAISE_TICKET:
            return self.ticket.create(query)
        elif intent == Intent.TRACK_TICKET:
            return self.ticket.track(entities.get("ticket_id", "TKT-000000"))
        elif intent == Intent.ESCALATE:
            return self.escalation.escalate(query)
        return "Unable to understand your query. Please rephrase or contact support."


if __name__ == "__main__":
    bot = ProcurementOrchestrator()
    queries = [
        "What is the EMD requirement for bid submission?",
        "What is the status of bid BID-2024-001?",
        "Show me details of order ORD-5002",
        "I want to raise a ticket about a delayed delivery",
        "I need to speak to a human agent",
    ]
    for q in queries:
        print(f"\n{'='*60}\nQ: {q}")
        print(f"A: {bot.handle(q)}")
