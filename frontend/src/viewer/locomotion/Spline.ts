type Point3 = [number, number, number];

/**
 * Catmull-Rom spline through a polyline of control points (interpolates each point exactly).
 * t in [0,1] is normalized over the whole spline; internally mapped to segment + local t.
 *
 * For 2 points: degenerates to linear interpolation.
 */
export class CatmullRomSpline {
  private readonly points: Point3[];

  constructor(points: Point3[]) {
    if (points.length < 2) {
      throw new Error("CatmullRomSpline needs >= 2 control points");
    }
    this.points = points;
  }

  position(t: number): Point3 {
    const tt = Math.max(0, Math.min(1, t));
    const n = this.points.length;
    if (n === 2) return this.lerp(this.points[0], this.points[1], tt);

    const seg = Math.min(Math.floor(tt * (n - 1)), n - 2);
    const local = tt * (n - 1) - seg;
    const p0 = this.points[Math.max(0, seg - 1)];
    const p1 = this.points[seg];
    const p2 = this.points[seg + 1];
    const p3 = this.points[Math.min(n - 1, seg + 2)];
    return this.catmullRom(p0, p1, p2, p3, local);
  }

  tangent(t: number, eps = 1e-3): Point3 {
    const a = this.position(Math.max(0, t - eps));
    const b = this.position(Math.min(1, t + eps));
    return [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  }

  private lerp(a: Point3, b: Point3, t: number): Point3 {
    return [
      a[0] + (b[0] - a[0]) * t,
      a[1] + (b[1] - a[1]) * t,
      a[2] + (b[2] - a[2]) * t,
    ];
  }

  private catmullRom(p0: Point3, p1: Point3, p2: Point3, p3: Point3, t: number): Point3 {
    const t2 = t * t;
    const t3 = t2 * t;
    const out: Point3 = [0, 0, 0];
    for (let i = 0; i < 3; i++) {
      out[i] = 0.5 * (
        2 * p1[i] +
        (-p0[i] + p2[i]) * t +
        (2 * p0[i] - 5 * p1[i] + 4 * p2[i] - p3[i]) * t2 +
        (-p0[i] + 3 * p1[i] - 3 * p2[i] + p3[i]) * t3
      );
    }
    return out;
  }
}
