import { describe, expect, it } from "vitest";

import { CatmullRomSpline } from "./Spline";

const points: [number, number, number][] = [
  [0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 1, 0],
];

describe("CatmullRomSpline", () => {
  it("returns first point at t=0", () => {
    const s = new CatmullRomSpline(points);
    const [x, y, z] = s.position(0);
    expect(x).toBeCloseTo(0);
    expect(y).toBeCloseTo(0);
    expect(z).toBeCloseTo(0);
  });

  it("returns last point at t=1", () => {
    const s = new CatmullRomSpline(points);
    const [x, y, z] = s.position(1);
    expect(x).toBeCloseTo(3);
    expect(y).toBeCloseTo(1);
    expect(z).toBeCloseTo(0);
  });

  it("interpolates monotonically in x", () => {
    const s = new CatmullRomSpline(points);
    expect(s.position(0.5)[0]).toBeGreaterThan(s.position(0.25)[0]);
  });

  it("two-point spline reduces to linear interpolation", () => {
    const s = new CatmullRomSpline([[0, 0, 0], [10, 0, 0]]);
    expect(s.position(0.5)[0]).toBeCloseTo(5);
  });

  it("throws on fewer than 2 points", () => {
    expect(() => new CatmullRomSpline([[0, 0, 0]])).toThrow();
  });

  it("tangent points roughly forward along x for monotonic spline", () => {
    const s = new CatmullRomSpline(points);
    const [dx, , dz] = s.tangent(0.5);
    expect(dx).toBeGreaterThan(0);
    expect(Math.abs(dz)).toBeLessThan(0.5);
  });
});
