export interface MouseLookDelta {
  yawDelta: number;   // radians
  pitchDelta: number;
}

export class MouseLook {
  private yaw = 0;
  private pitch = 0;
  private active = false;
  private readonly element: HTMLElement;

  constructor(element: HTMLElement) {
    this.element = element;
    element.addEventListener("click", this.onClick);
    document.addEventListener("pointerlockchange", this.onLockChange);
    document.addEventListener("mousemove", this.onMove);
  }

  delta(): MouseLookDelta {
    return { yawDelta: this.yaw, pitchDelta: this.pitch };
  }

  reset(yaw = 0, pitch = 0): void {
    this.yaw = yaw;
    this.pitch = pitch;
  }

  dispose(): void {
    this.element.removeEventListener("click", this.onClick);
    document.removeEventListener("pointerlockchange", this.onLockChange);
    document.removeEventListener("mousemove", this.onMove);
  }

  private onClick = () => this.element.requestPointerLock();
  private onLockChange = () => {
    this.active = document.pointerLockElement === this.element;
  };
  private onMove = (e: MouseEvent) => {
    if (!this.active) return;
    const sens = 0.002;
    this.yaw -= e.movementX * sens;
    this.pitch -= e.movementY * sens;
    const limit = Math.PI / 2 - 0.01;
    this.pitch = Math.max(-limit, Math.min(limit, this.pitch));
  };
}
