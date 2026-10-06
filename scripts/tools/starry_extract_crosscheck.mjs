// Runs starry-digitizer's own Symbol Extract / Line Extract (MIT, (c) 2021
// MATO Tomoya) on the cases written by gen_starry_extract_fixtures.py, so the
// Python port (domain/starry_extract.py) can be checked point for point.
//
// The classes are cut out of the compiled library (library-build/dist/core.js
// of the vendored package) from the base class with matchColor up to the
// Extractor class, and evaluated as they are -- nothing is rewritten.
//
//   node starry_extract_crosscheck.mjs <core.js> <cases.json>  -> JSON on stdout
import { readFileSync } from "node:fs";

const [corePath, casesPath] = process.argv.slice(2);
const core = readFileSync(corePath, "utf8");
const start = core.search(/class \w+ \{\n\s*matchColor\(/);
const end = core.indexOf('n(this, "strategies", ["Symbol Extract", "Line Extract"])');
if (start < 0 || end < 0) throw new Error("extract strategies not found in " + corePath);
const endClass = core.lastIndexOf("\nclass ", end);
const src = core.slice(start, endClass);
const names = [...src.matchAll(/^class (\w+)/gm)].map((m) => m[1]);
// `n` is the bundle's defineProperty helper
const factory = new Function("n", src + "\nreturn {" + names.join(",") + "};");
const classes = factory((obj, key, value) => {
  Object.defineProperty(obj, key, { value, enumerable: true, configurable: true, writable: true });
  return value;
});
const byName = {};
for (const C of Object.values(classes)) {
  const inst = new C();
  if (inst.name) byName[inst.name] = C;
}

const cases = JSON.parse(readFileSync(casesPath, "utf8"));
const out = cases.map((c) => {
  const s = new byName[c.strategy]();
  if (c.strategy === "Symbol Extract") {
    s.setMinDiameterPx(c.params.min_diameter_px);
    s.setMaxDiameterPx(c.params.max_diameter_px);
  } else {
    s.setDxPx(c.params.dx_px);
    s.setDyPx(c.params.dy_px);
  }
  const pts = s.execute(
    c.height, c.width, c.rgba, c.mask_rgba ?? [], !!c.mask_rgba, c.target, c.distance_pct,
  );
  return pts.map((p) => [p.xPx, p.yPx]);
});
process.stdout.write(JSON.stringify(out));
