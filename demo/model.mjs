import weights from "./coefficients.mjs";

// Same basis and 365-day calendar as the synthetic training experiment.
export function simulate(day, ndvi, ndwi) {
  const angle = 2 * Math.PI * (day - 1) / 365;
  const [bias, sin, cos, vegetation, water] = weights;
  return Math.max(0, bias + sin * Math.sin(angle) + cos * Math.cos(angle)
    + vegetation * ndvi + water * ndwi);
}

export function dayLabel(day) {
  const date = new Date(Date.UTC(2021, 0, day));
  const dd = String(date.getUTCDate()).padStart(2, "0");
  const mm = String(date.getUTCMonth() + 1).padStart(2, "0");
  return dd + "/" + mm;
}
