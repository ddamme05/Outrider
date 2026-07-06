"""Invoice math: exact-decimal line items, tax, and discounts.

`Invoice` accumulates `LineItem` rows and derives the customer-facing amounts —
per-line extended price, subtotal, tax, and the discounted grand total — using
`decimal.Decimal` throughout so money never rounds through a binary float. Every
derived amount is quantized to whole cents with half-up rounding, matching the
ledger the billing service reconciles against.

Quantities and unit prices arrive from the order-entry service and are stored
as-is on each line; the derivation methods read them back when a caller asks for
a total.
"""

from __future__ import annotations

import json

from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal

# Money is carried to whole cents; every public amount quantizes to this.
CENTS = Decimal("0.01")


@dataclass
class LineItem:
    """A single billable row on an invoice."""

    sku: str
    description: str
    quantity: int
    unit_price: Decimal


@dataclass
class Invoice:
    """A customer invoice: line items plus tax and discount policy."""

    currency: str = "USD"
    tax_rate: Decimal = Decimal("0")
    items: list[LineItem] = field(default_factory=list)
    _discount: Decimal = field(default=Decimal("0.00"))

    def add_line(
        self,
        sku: str,
        description: str,
        quantity: int,
        unit_price: str,
    ) -> LineItem:
        """Append a line item, storing the price as an exact Decimal."""
        item = LineItem(
            sku=sku,
            description=description,
            quantity=quantity,
            unit_price=Decimal(unit_price),
        )
        self.items.append(item)
        return item

    def line_total(self, quantity, unit_price):
        """Extended price for one line: quantity times unit price, to the cent."""
        extended = Decimal(quantity) * Decimal(unit_price)
        return extended.quantize(CENTS, rounding=ROUND_HALF_UP)

    def subtotal(self) -> Decimal:
        """Sum of every line's extended price."""
        running = Decimal("0.00")
        for item in self.items:
            running += self.line_total(item.quantity, item.unit_price)
        return running.quantize(CENTS, rounding=ROUND_HALF_UP)

    def tax(self) -> Decimal:
        """Tax owed on the subtotal at the invoice's configured rate."""
        return (self.subtotal() * self.tax_rate).quantize(CENTS, rounding=ROUND_HALF_UP)

    def apply_discount(self, amount: str) -> Decimal:
        """Set a flat discount subtracted from the taxed total."""
        self._discount = Decimal(amount)
        return self._discount

    def total(self) -> Decimal:
        """Grand total: subtotal plus tax, less any applied discount."""
        gross = self.subtotal() + self.tax()
        return (gross - self._discount).quantize(CENTS, rounding=ROUND_HALF_UP)
