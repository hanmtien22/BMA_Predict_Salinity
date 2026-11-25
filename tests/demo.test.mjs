import assert from "node:assert/strict";
import test from "node:test";
import { simulate, dayLabel } from "../demo/model.mjs";

test("reference scenario, yearly cycle and date boundaries", () => {
  const value = simulate(105, 0.4, -0.1);
  assert.ok(value > 14 && value < 16);
  assert.equal(simulate(105, 0.4, -0.1), value);
  assert.ok(Math.abs(simulate(470, 0.4, -0.1) - value) < 1e-12);
  assert.equal(dayLabel(1), "01/01");
  assert.equal(dayLabel(365), "31/12");
});

test("both spectral inputs affect results across the supported slider ranges", () => {
  for (const day of [1, 105, 365]) {
    assert.ok(simulate(day, -1, 0) > simulate(day, 1, 0));
    assert.ok(simulate(day, 0, -1) > simulate(day, 0, 1));
    for (const ndvi of [-1, 1]) {
      for (const ndwi of [-1, 1]) {
        assert.ok(Number.isFinite(simulate(day, ndvi, ndwi)));
        assert.ok(simulate(day, ndvi, ndwi) >= 0);
      }
    }
  }
});
