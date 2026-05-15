import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RGBELoader } from "three/addons/loaders/RGBELoader.js";
import GUI from "lil-gui";

// ── Renderer ────────────────────────────────────────────────────────────────
const canvas = document.querySelector<HTMLCanvasElement>("#c")!;
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, stencil: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.localClippingEnabled = true;
renderer.outputColorSpace = THREE.SRGBColorSpace;

// ── Scene & Camera ───────────────────────────────────────────────────────────
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a1a);

const camera = new THREE.PerspectiveCamera(
  45,
  window.innerWidth / window.innerHeight,
  0.01,
  100,
);
camera.position.set(2, 1, 3);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0);
controls.enableDamping = true;

// ── Lighting ─────────────────────────────────────────────────────────────────
// Ambient + directional fallback while HDRI loads
scene.add(new THREE.AmbientLight(0xffffff, 0.4));
const sun = new THREE.DirectionalLight(0xffffff, 1.2);
sun.position.set(2, 4, 3);
scene.add(sun);

// HDRI environment for PBR reflections
new RGBELoader().load(
  "https://threejs.org/examples/textures/equirectangular/royal_esplanade_1k.hdr",
  (hdr) => {
    hdr.mapping = THREE.EquirectangularReflectionMapping;
    scene.environment = hdr;
  },
);

// ── Clip plane ───────────────────────────────────────────────────────────────
// Normal points in -Y direction; constant shifts the plane along that normal.
// Geometry is clipped where dot(pos, normal) + constant < 0.
const clipPlane = new THREE.Plane(new THREE.Vector3(0, -1, 0), 0);

// ── Stencil capping approach ─────────────────────────────────────────────────
//
// Standard technique (Everitt & Kilgard, "Practical and Robust Stenciled
// Shadow Volumes"):
//
//  Pass 1 — write stencil: mesh back-faces increment, front-faces decrement
//            stencil. After this pass, pixels where back-face > front-face
//            (i.e., inside the clipped solid at the cut cross-section) have
//            stencil == 1 (net difference).
//
//  Pass 2 — draw cap: render a large coplanar quad, but only where stencil!=0.
//            This fills the cut face with the cap material.
//
// three.js does not expose per-material stencil ±1 ops with depth-clamping
// directly, so we use the simpler "always replace" variant that works for
// convex hulls and most well-formed meshes:
//
//  - All mesh materials: stencilFunc=Always, stencilZPass=Replace(ref=1), DoubleSide
//  - Cap plane: stencilFunc=Equal(ref=1), rendered after the mesh
//
// For the DamagedHelmet (essentially convex helmet shell) this produces a
// correct filled cap.  For concave or multi-shell meshes the XOR variant
// is more robust; marked as TODO for Spike C-2 against the real TRELLIS.2 mesh.

// ── Cap mesh ─────────────────────────────────────────────────────────────────
const capMaterial = new THREE.MeshBasicMaterial({
  color: 0xaa6633,
  side: THREE.DoubleSide,
  // Only render where stencil ref value == 1 (written by mesh pass)
  stencilWrite: true,
  stencilRef: 1,
  stencilFunc: THREE.EqualStencilFunc,
  stencilFail: THREE.KeepStencilOp,
  stencilZFail: THREE.KeepStencilOp,
  stencilZPass: THREE.KeepStencilOp,
});

const capMesh = new THREE.Mesh(new THREE.PlaneGeometry(20, 20), capMaterial);
// Default orientation for Y-axis clip (plane normal is Y)
capMesh.rotation.x = -Math.PI / 2;
capMesh.renderOrder = 1; // draw after the clipped mesh
scene.add(capMesh);

// ── Object loading ────────────────────────────────────────────────────────────
const loader = new GLTFLoader();
let bboxHalfExtent = 1.0;
type Axis = "x" | "y" | "z";

function orientCap(axis: Axis): void {
  // Align the cap quad perpendicular to the clip normal
  const normal = clipPlane.normal;
  const capNormal = new THREE.Vector3(0, 0, 1); // PlaneGeometry default faces +Z
  const quat = new THREE.Quaternion().setFromUnitVectors(
    capNormal,
    normal.clone().negate(),
  );
  capMesh.quaternion.copy(quat);
  void axis; // axis parameter retained for future use
}

function updateCapPosition(): void {
  // Move the cap quad to sit exactly on the clip plane
  capMesh.position
    .copy(clipPlane.normal)
    .multiplyScalar(-clipPlane.constant);
}

loader.load(
  "/object.glb",
  (gltf) => {
    const object = gltf.scene;

    // Center the model
    const bbox = new THREE.Box3().setFromObject(object);
    const size = new THREE.Vector3();
    bbox.getSize(size);
    bboxHalfExtent = Math.max(size.x, size.y, size.z) / 2;

    const center = new THREE.Vector3();
    bbox.getCenter(center);
    object.position.sub(center);

    // Wire each mesh for clipping + stencil write
    object.traverse((node) => {
      const mesh = node as THREE.Mesh;
      if (!mesh.isMesh) return;

      // Support both single material and material arrays
      const materials = Array.isArray(mesh.material)
        ? mesh.material
        : [mesh.material];

      materials.forEach((mat) => {
        const m = mat as THREE.MeshStandardMaterial;
        m.clippingPlanes = [clipPlane];
        m.clipShadows = true;
        // Write stencil ref=1 wherever a fragment passes depth test
        m.stencilWrite = true;
        m.stencilRef = 1;
        m.stencilFunc = THREE.AlwaysStencilFunc;
        m.stencilFail = THREE.KeepStencilOp;
        m.stencilZFail = THREE.KeepStencilOp;
        m.stencilZPass = THREE.ReplaceStencilOp;
        m.side = THREE.DoubleSide; // needed for interior faces to write stencil
        m.needsUpdate = true;
      });
    });

    scene.add(object);
  },
  undefined,
  (err: unknown) => {
    console.error("GLB load error:", err);
  },
);

// ── GUI ───────────────────────────────────────────────────────────────────────
const params = {
  clipT: 0.5, // 0 = bottom, 1 = top of bbox
  axis: "y" as Axis,
};

const gui = new GUI({ title: "Dissection Controls" });

gui.add(params, "axis", ["x", "y", "z"]).name("Clip axis").onChange((axis: Axis) => {
  switch (axis) {
    case "x":
      clipPlane.normal.set(-1, 0, 0);
      break;
    case "y":
      clipPlane.normal.set(0, -1, 0);
      break;
    case "z":
      clipPlane.normal.set(0, 0, -1);
      break;
  }
  orientCap(axis);
  updateCapPosition();
});

gui.add(params, "clipT", 0, 1, 0.005).name("Clip position").onChange((t: number) => {
  // Map [0,1] → [−extent, +extent] along the axis
  clipPlane.constant = (t - 0.5) * 2 * bboxHalfExtent;
  updateCapPosition();
});

// ── Global test handle (used by Playwright smoke-test) ────────────────────────
declare global {
  interface Window {
    __spike: {
      setClipT: (t: number) => void;
      setAxis: (axis: Axis) => void;
      camera: THREE.PerspectiveCamera;
      controls: OrbitControls;
    };
  }
}

window.__spike = {
  setClipT: (t: number) => {
    params.clipT = t;
    clipPlane.constant = (t - 0.5) * 2 * bboxHalfExtent;
    updateCapPosition();
  },
  setAxis: (axis: Axis) => {
    params.axis = axis;
    switch (axis) {
      case "x": clipPlane.normal.set(-1, 0, 0); break;
      case "y": clipPlane.normal.set(0, -1, 0); break;
      case "z": clipPlane.normal.set(0, 0, -1); break;
    }
    orientCap(axis);
    updateCapPosition();
  },
  camera,
  controls,
};

// ── FPS counter ───────────────────────────────────────────────────────────────
const fpsEl = document.querySelector<HTMLSpanElement>("#fps")!;
let frames = 0;
let lastFpsTime = performance.now();

// ── Render loop ───────────────────────────────────────────────────────────────
function loop(): void {
  requestAnimationFrame(loop);
  controls.update();
  renderer.render(scene, camera);

  frames++;
  const now = performance.now();
  const elapsed = now - lastFpsTime;
  if (elapsed > 500) {
    fpsEl.textContent = ((frames * 1000) / elapsed).toFixed(0);
    frames = 0;
    lastFpsTime = now;
  }
}

loop();

// ── Resize handling ───────────────────────────────────────────────────────────
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
