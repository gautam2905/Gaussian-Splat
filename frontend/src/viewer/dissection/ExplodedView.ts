import * as THREE from "three";

interface PartState {
  object: THREE.Object3D;
  originalPos: THREE.Vector3;
  outwardDir: THREE.Vector3;
}

/**
 * Animates a radial "exploded view" of a multi-component GLB scene.
 *
 * Each direct mesh child (or first-level group child) of the root is treated
 * as a separable part. Parts are offset radially from the object's centroid
 * proportional to explodeT.
 *
 * Returns false from attach() when the root has fewer than 2 separable parts,
 * so the caller can disable the exploded-view slider.
 */
export class ExplodedView {
  private parts: PartState[] = [];
  private maxDistance = 0;
  private attached = false;

  /**
   * Returns true if the object has 2+ separable components (explode available).
   * Returns false for single-mesh objects.
   */
  attach(root: THREE.Object3D): boolean {
    this.detach();

    // First pass: direct mesh children
    const candidates: THREE.Object3D[] = [];
    for (const child of root.children) {
      if ((child as THREE.Mesh).isMesh || child.children.length > 0) {
        candidates.push(child);
      }
    }

    // Second pass: if top level has only one child (common in TRELLIS.2 GLBs
    // where scene → group → meshes), go one level deeper.
    if (candidates.length < 2 && root.children.length === 1) {
      const inner = root.children[0];
      for (const child of inner.children) {
        if ((child as THREE.Mesh).isMesh || child.children.length > 0) {
          candidates.push(child);
        }
      }
    }

    if (candidates.length < 2) return false;

    const center = new THREE.Vector3();
    new THREE.Box3().setFromObject(root).getCenter(center);

    this.maxDistance = 0;
    this.parts = candidates.map((obj) => {
      const partCenter = new THREE.Vector3();
      new THREE.Box3().setFromObject(obj).getCenter(partCenter);
      const dir = new THREE.Vector3().subVectors(partCenter, center);
      const dist = dir.length();
      if (dist > 1e-6) dir.normalize();
      else dir.set(1, 0, 0); // degenerate: push along X as fallback
      this.maxDistance = Math.max(this.maxDistance, dist);
      return {
        object: obj,
        originalPos: obj.position.clone(),
        outwardDir: dir,
      };
    });

    this.attached = true;
    return true;
  }

  /**
   * t in [0, 1]. scale controls how far parts travel relative to their natural
   * centroid distance (default 1.5 = 50% further than their own radius).
   */
  setT(t: number, scale = 1.5): void {
    if (!this.attached) return;
    const clamped = Math.max(0, Math.min(1, t));
    for (const p of this.parts) {
      const offset = p.outwardDir
        .clone()
        .multiplyScalar(clamped * scale * this.maxDistance);
      p.object.position.copy(p.originalPos).add(offset);
    }
  }

  detach(): void {
    for (const p of this.parts) {
      p.object.position.copy(p.originalPos);
    }
    this.parts = [];
    this.maxDistance = 0;
    this.attached = false;
  }
}
