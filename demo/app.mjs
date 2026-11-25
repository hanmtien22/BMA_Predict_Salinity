import { dayLabel, simulate } from "./model.mjs";

const byId = id => document.getElementById(id);
const format = digits => new Intl.NumberFormat("vi-VN", {
  minimumFractionDigits: digits, maximumFractionDigits: digits
});
const index = format(2);
const salinity = format(1);
const x = day => 38 + (day - 1) * 470 / 364;
const y = value => 164 - value * 150 / 24;

function render() {
  const day = byId("day").valueAsNumber;
  const ndvi = byId("ndvi").valueAsNumber;
  const ndwi = byId("ndwi").valueAsNumber;
  const value = simulate(day, ndvi, ndwi);
  const date = dayLabel(day);
  byId("date-text").textContent = date;
  byId("day").setAttribute("aria-valuetext", date);
  byId("ndvi-text").textContent = index.format(ndvi);
  byId("ndwi-text").textContent = index.format(ndwi);
  byId("value").textContent = salinity.format(value);
  const coords = Array.from({length: 365}, (_, i) =>
    x(i + 1).toFixed(2) + " " + y(simulate(i + 1, ndvi, ndwi)).toFixed(2));
  const line = "M" + coords.join(" L");
  byId("curve").setAttribute("d", line);
  byId("area").setAttribute("d", line + " L508 164 H38 Z");
  byId("guide").setAttribute("d", "M" + x(day) + " 164 V" + y(value));
  byId("point").setAttribute("cx", x(day));
  byId("point").setAttribute("cy", y(value));
  byId("chart").setAttribute("aria-label",
    "Chu kỳ năm tại NDVI " + index.format(ndvi) + " và NDWI " + index.format(ndwi)
    + ". Ngày " + date + " đạt " + salinity.format(value) + " phần nghìn.");
}

byId("controls").addEventListener("input", render);
byId("controls").addEventListener("reset", () => requestAnimationFrame(render));
render();
