import * as THREE from "three";

/**
 * Manages stencil-buffer capped clipping for a focused object.
 *
 * Two-pass technique (proven in Spike C at 144 fps):
 *   Pass 1 — mesh materials: stencilFunc=Always, stencilZPass=Replace(ref=1), DoubleSide
 *             Writes stencil ref=1 wherever a fragment passes the depth test.
 *   Pass 2 — cap quad: stencilFunc=Equal(ref=1), drawn after the mesh.
 *             Only pixels where stencil==1 (i.e. inside the cut cross-section) are filled.
 *
 * Works correctly for convex and most real-world meshes. For concave or
 * multi-shell meshes the XOR variant is more robust (future work).
 *
 * Prerequisites (already set in ExhibitViewer.tsx by Phase 3):
 *   renderer constructed with { stencil: true }
 *   renderer.localClippingEnabled = true
 *
 * On detach, every touched material is restored to its state before attach,
 * so the object looks normal again when the user exits focus mode.
 */

type SavedMaterialState = {
  mesh: THREE.Mesh;
  material: THREE.Material;
  originalClippingPlanes: THREE.Plane[] | null;
  originalStencilWrite: boolean;
  originalStencilRef: number;
  originalStencilFunc: THREE.StencilFunc;
  originalStencilZPass: THREE.StencilOp;
  originalSide: THREE.Side;
};

export class CappedClipping {
  private readonly scene: THREE.Scene;
  private readonly capMesh: THREE.Mesh;
  private readonly capMaterial: THREE.MeshBasicMaterial;
  private saved: SavedMaterialState[] = [];
  private attached = false;

  constructor(scene: THREE.Scene) {
    this.scene = scene;
    this.capMaterial = new THREE.MeshBasicMaterial({
      color: 0x6b4a2a, // warm brass default; overwritten from object's material on attach
      side: THREE.DoubleSide,
      stencilWrite: true,
      stencilRef: 1,
      stencilFunc: THREE.EqualStencilFunc,
      stencilFail: THREE.KeepStencilOp,
      stencilZFail: THREE.KeepStencilOp,
      stencilZPass: THREE.KeepStencilOp,
    });

    this.capMesh = new THREE.Mesh(new THREE.PlaneGeometry(20, 20), this.capMaterial);
    this.capMesh.renderOrder = 1; // draw after the clipped object
    this.capMesh.visible = false;
    scene.add(this.capMesh);
  }

  attach(object: THREE.Object3D, plane: THREE.Plane): void {
    if (this.attached) this.detach();
    this.attached = true;

    // Sample the object's primary color for the cap face so it looks contiguous.
    let sampledColor: THREE.Color | null = null;
    object.traverse((child) => {
      if (sampledColor) return;
      const mesh = child as THREE.Mesh;
      if (mesh.isMesh) {
        const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
        for (const mat of mats) {
          const m = mat as THREE.MeshStandardMaterial;
          if (m.color) {
            sampledColor = m.color.clone();
            break;
          }
        }
      }
    });
    if (sampledColor) this.capMaterial.color.copy(sampledColor);

    // Wire stencil writes on every material of the focused object's mesh tree.
    object.traverse((child) => {
      const mesh = child as THREE.Mesh;
      if (!mesh.isMesh) return;

      const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
      for (const mat of mats) {
        const m = mat as THREE.MeshStandardMaterial;
        this.saved.push({
          mesh,
          material: m,
          originalClippingPlanes: m.clippingPlanes ? m.clippingPlanes.slice() : null,
          originalStencilWrite: m.stencilWrite,
          originalStencilRef: m.stencilRef,
          originalStencilFunc: m.stencilFunc as THREE.StencilFunc,
          originalStencilZPass: m.stencilZPass as THREE.StencilOp,
          originalSide: m.side,
        });
        m.clippingPlanes = [plane];
        m.stencilWrite = true;
        m.stencilRef = 1;
        m.stencilFunc = THREE.AlwaysStencilFunc;
        m.stencilZPass = THREE.ReplaceStencilOp;
        m.side = THREE.DoubleSide;
        m.needsUpdate = true;
      }
    });

    this.capMesh.visible = true;
  }

  /**
   * Call every frame while in focus mode.
   * Positions and orients the cap quad to lie exactly on the clip plane.
   */
  update(plane: THREE.Plane): void {
    if (!this.attached) return;

    // Position: point on plane = normal * (−constant)
    this.capMesh.position
      .copy(plane.normal)
      .multiplyScalar(-plane.constant);

    // Orient: cap quad's local +Z → negated clip normal (so the visible face
    // points back toward the viewer through the cut).
    const target = plane.normal.clone().negate();
    const quat = new THREE.Quaternion().setFromUnitVectors(
      new THREE.Vector3(0, 0, 1),
      target,
    );
    this.capMesh.quaternion.copy(quat);
  }

  detach(): void {
    if (!this.attached) return;
    for (const s of this.saved) {
      const m = s.material as THREE.MeshStandardMaterial;
      m.clippingPlanes = s.originalClippingPlanes;
      m.stencilWrite = s.originalStencilWrite;
      m.stencilRef = s.originalStencilRef;
      m.stencilFunc = s.originalStencilFunc;
      m.stencilZPass = s.originalStencilZPass;
      m.side = s.originalSide;
      m.needsUpdate = true;
    }
    this.saved = [];
    this.capMesh.visible = false;
    this.attached = false;
  }

  dispose(): void {
    this.detach();
    this.scene.remove(this.capMesh);
    this.capMaterial.dispose();
    this.capMesh.geometry.dispose();
  }
}
