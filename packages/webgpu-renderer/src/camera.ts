/**
 * Simple Z-up orbit camera — no dependencies, pure Float32Array math.
 * Column-major matrices for WebGPU.
 *
 * Spherical coordinates: alpha (azimuth), beta (polar), radius.
 * Camera position = target + sphericalToCartesian(alpha, beta, radius).
 */
import { mat4Inverse } from './math';

export class OrbitCamera {
  target: Float64Array = new Float64Array([0, 0, 0]);
  alpha = Math.PI / 4;
  beta = Math.PI / 4;
  radius = 150;

  near = 0.1;
  far = 10000;
  fov = Math.PI / 3;

  /** Column-major 4x4 view-projection matrix, written every frame. */
  viewProj = new Float32Array(16);
  /** Column-major 4x4 view-projection matrix from the previous frame. */
  prevViewProj = new Float32Array(16);
  /** Column-major 4x4 inverse of current view-projection, for velocity pass. */
  invViewProj = new Float32Array(16);
  /** Column-major 4x4 view matrix. */
  viewMat = new Float32Array(16);
  /** Column-major 4x4 projection matrix, written every frame. */
  proj = new Float32Array(16);
  /** Camera world position, written every frame. */
  position = new Float64Array([0, 0, 0]);

  // Interaction state
  private _isDragging = false;
  private _lastX = 0;
  private _lastY = 0;
  private _pinchDist = 0;

  attach(canvas: HTMLCanvasElement): () => void {
    const onDown = (e: MouseEvent | TouchEvent) => {
      if (e instanceof MouseEvent) {
        this._isDragging = true;
        this._lastX = e.clientX;
        this._lastY = e.clientY;
      } else if ('touches' in e && e.touches.length >= 1) {
        this._isDragging = true;
        this._lastX = e.touches[0].clientX;
        this._lastY = e.touches[0].clientY;
        if (e.touches.length === 2) {
          const dx = e.touches[0].clientX - e.touches[1].clientX;
          const dy = e.touches[0].clientY - e.touches[1].clientY;
          this._pinchDist = Math.sqrt(dx * dx + dy * dy);
        }
      }
    };
    const onMove = (e: MouseEvent | TouchEvent) => {
      if (!this._isDragging) return;
      let cx: number, cy: number;
      if (e instanceof MouseEvent) {
        cx = e.clientX; cy = e.clientY;
      } else if ('touches' in e && e.touches.length >= 1) {
        cx = e.touches[0].clientX; cy = e.touches[0].clientY;
      } else return;

      if ('touches' in e && e.touches.length === 2) {
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (this._pinchDist > 0) {
          this.radius *= this._pinchDist / dist;
          this.radius = Math.max(1, Math.min(this.radius, 10000));
        }
        this._pinchDist = dist;
        return;
      }

      const dx = cx - this._lastX;
      const dy = cy - this._lastY;

      if (e instanceof MouseEvent && e.buttons === 2) {
        // Right-click pan
        const fwdX = this.target[0] - this.position[0];
        const fwdY = this.target[1] - this.position[1];
        const len = Math.sqrt(fwdX * fwdX + fwdY * fwdY);
        if (len > 0.001) {
          const panSpeed = this.radius * 0.002;
          this.target[0] += (-fwdY / len) * (-dx * panSpeed);
          this.target[1] += (fwdX / len) * (-dx * panSpeed);
          this.target[2] += dy * panSpeed;
        }
      } else {
        this.alpha -= dx * 0.005;
        this.beta = Math.max(0.01, Math.min(Math.PI - 0.01, this.beta - dy * 0.005));
      }

      this._lastX = cx;
      this._lastY = cy;
    };
    const onUp = () => { this._isDragging = false; };

    canvas.addEventListener('mousedown', onDown);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    canvas.addEventListener('wheel', (e) => {
      this.radius *= e.deltaY > 0 ? 1.08 : 0.92;
      this.radius = Math.max(1, Math.min(this.radius, 10000));
    });
    canvas.addEventListener('touchstart', onDown, { passive: true });
    window.addEventListener('touchmove', onMove, { passive: true });
    window.addEventListener('touchend', onUp);

    return () => {
      canvas.removeEventListener('mousedown', onDown);
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      canvas.removeEventListener('wheel', onUp);
      canvas.removeEventListener('touchstart', onDown);
      window.removeEventListener('touchmove', onMove);
      window.removeEventListener('touchend', onUp);
    };
  }

  /** Recompute view-projection matrix. Call once per frame. */
  update(aspect: number) {
    const sinB = Math.sin(this.beta);
    const cosB = Math.cos(this.beta);
    const sinA = Math.sin(this.alpha);
    const cosA = Math.cos(this.alpha);

    this.position[0] = this.target[0] + this.radius * sinB * cosA;
    this.position[1] = this.target[1] + this.radius * sinB * sinA;
    this.position[2] = this.target[2] + this.radius * cosB;

    // Save previous frame's view-projection before computing new one
    this.prevViewProj.set(this.viewProj);

    // View matrix: lookAt(position, target, up=Z)
    const fwd = [
      this.target[0] - this.position[0],
      this.target[1] - this.position[1],
      this.target[2] - this.position[2],
    ];
    const fwdLen = Math.sqrt(fwd[0] * fwd[0] + fwd[1] * fwd[1] + fwd[2] * fwd[2]);
    if (fwdLen > 0) { fwd[0] /= fwdLen; fwd[1] /= fwdLen; fwd[2] /= fwdLen; }

    // right = fwd × up (0,0,1)
    const right = [fwd[1], -fwd[0], 0];
    const rightLen = Math.sqrt(right[0] * right[0] + right[1] * right[1]);
    if (rightLen > 0.001) { right[0] /= rightLen; right[1] /= rightLen; }
    else { right[0] = 1; right[1] = 0; }

    // up = right × fwd
    const up = [
      right[1] * fwd[2] - right[2] * fwd[1],
      right[2] * fwd[0] - right[0] * fwd[2],
      right[0] * fwd[1] - right[1] * fwd[0],
    ];

    // Column-major 4x4 view matrix
    const view = [
      right[0], up[0], -fwd[0], 0,
      right[1], up[1], -fwd[1], 0,
      right[2], up[2], -fwd[2], 0,
      -(right[0] * this.position[0] + right[1] * this.position[1] + right[2] * this.position[2]),
      -(up[0] * this.position[0] + up[1] * this.position[1] + up[2] * this.position[2]),
      fwd[0] * this.position[0] + fwd[1] * this.position[1] + fwd[2] * this.position[2],
      1,
    ];
    this.viewMat.set(view);

    // Perspective projection (column-major)
    const f = 1 / Math.tan(this.fov / 2);
    const nf = 1 / (this.near - this.far);
    const proj = [
      f / aspect, 0, 0, 0,
      0, f, 0, 0,
      0, 0, this.far * nf, -1,
      0, 0, this.far * this.near * nf, 0,
    ];
    this.proj.set(proj);

    // Multiply: viewProj = proj × view
    const vp = new Float32Array(16);
    for (let col = 0; col < 4; col++) {
      for (let row = 0; row < 4; row++) {
        vp[col * 4 + row] =
          proj[row] * view[col * 4] +
          proj[4 + row] * view[col * 4 + 1] +
          proj[8 + row] * view[col * 4 + 2] +
          proj[12 + row] * view[col * 4 + 3];
      }
    }
    this.viewProj.set(vp);
    mat4Inverse(this.viewProj, this.invViewProj);
  }
}
