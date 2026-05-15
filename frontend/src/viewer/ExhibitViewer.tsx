import { useEffect, useRef } from "react";
import * as THREE from "three";

import type { Manifest } from "../api/types";
import { CameraStateMachine } from "./locomotion/CameraStateMachine";
import { MouseLook } from "./locomotion/MouseLook";
import { RailController } from "./locomotion/RailController";
import { loadScene } from "./Scene";
import { mountPedestals } from "./Pedestals";
import { useLocomotionStore } from "./store/locomotionStore";

interface Props {
  manifest: Manifest;
  apiBase: string;
}

export default function ExhibitViewer({ manifest, apiBase }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;

    // ===== three.js setup =====
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, stencil: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.localClippingEnabled = true; // ready for Phase 4 dissection

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(manifest.scene.ambient_light.color);
    scene.add(
      new THREE.AmbientLight(
        manifest.scene.ambient_light.color,
        manifest.scene.ambient_light.intensity,
      ),
    );
    const dir = new THREE.DirectionalLight(0xffffff, 0.6);
    dir.position.set(2, 4, 3);
    scene.add(dir);

    const camera = new THREE.PerspectiveCamera(
      60,
      window.innerWidth / window.innerHeight,
      0.05,
      100,
    );

    // ===== Assets =====
    const sceneHandle = loadScene(manifest.scene.scene_url, scene, apiBase);
    const pedestalsHandle = mountPedestals(manifest, scene, apiBase);

    // ===== Camera FSM =====
    const rail = new RailController(manifest);
    const mouseLook = new MouseLook(canvas);
    const fsm = new CameraStateMachine(camera, rail, mouseLook);

    // ===== Input handlers =====
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight" || e.key === " " || e.key === "ArrowDown") {
        e.preventDefault();
        fsm.inputAdvance();
      } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
        e.preventDefault();
        fsm.inputRetreat();
      } else if (e.key === "Escape") {
        fsm.exitFocus();
      }
    };

    const raycaster = new THREE.Raycaster();
    const ndc = new THREE.Vector2();
    const onClick = (e: MouseEvent) => {
      if (useLocomotionStore.getState().mode !== "walking") return;
      ndc.x = (e.clientX / window.innerWidth) * 2 - 1;
      ndc.y = -(e.clientY / window.innerHeight) * 2 + 1;
      raycaster.setFromCamera(ndc, camera);
      const hits = raycaster.intersectObjects(pedestalsHandle.group.children, true);
      for (const hit of hits) {
        let cur: THREE.Object3D | null = hit.object;
        while (cur) {
          const oid = cur.userData?.objectId as string | undefined;
          const pid = cur.userData?.pedestalId as string | undefined;
          if (oid && pid) {
            const mesh = pedestalsHandle.objectMeshes.get(oid);
            if (mesh) {
              fsm.enterFocus({ objectId: oid, pedestalId: pid, mesh });
              return;
            }
          }
          cur = cur.parent;
        }
      }
    };

    window.addEventListener("keydown", onKey);
    canvas.addEventListener("click", onClick);

    // ===== Render loop =====
    let raf = 0;
    const loop = () => {
      raf = requestAnimationFrame(loop);
      fsm.update(performance.now());
      renderer.render(scene, camera);
    };
    loop();

    // ===== Resize =====
    const onResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    };
    window.addEventListener("resize", onResize);

    // ===== Cleanup =====
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("keydown", onKey);
      canvas.removeEventListener("click", onClick);
      window.removeEventListener("resize", onResize);
      sceneHandle.dispose();
      pedestalsHandle.dispose();
      mouseLook.dispose();
      renderer.dispose();
    };
  }, [manifest, apiBase]);

  return <canvas ref={canvasRef} className="block" />;
}
