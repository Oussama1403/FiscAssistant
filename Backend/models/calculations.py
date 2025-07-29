from pydantic import BaseModel
import re

class VatInput(BaseModel):
    amount: float
    rate: float = 19.0

class ProfitInput(BaseModel):
    revenue: float
    expenses: float
    tax_rate: float = 0.15

def calculate_vat(amount: float, rate: float = 19.0) -> float:
    return amount * (rate / 100)

def calculate_net_profit(revenue: float, expenses: float, tax_rate: float = 0.15) -> float:
    taxable_income = revenue - expenses
    taxes = taxable_income * tax_rate
    return taxable_income - taxes

def parse_vat_input(message: str) -> VatInput:
    match = re.search(r"(\d+\.?\d*)\s*TND.*(\d+\.?\d*)%", message)
    if match:
        return VatInput(amount=float(match.group(1)), rate=float(match.group(2)))
    return None

def parse_profit_input(message: str) -> ProfitInput:
    match = re.search(r"revenue\s*(\d+\.?\d*)\s*TND.*expenses\s*(\d+\.?\d*)\s*TND", message, re.IGNORECASE)
    if match:
        return ProfitInput(revenue=float(match.group(1)), expenses=float(match.group(2)))
    return None