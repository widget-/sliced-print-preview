import './style.css';

async function main() {
  const canvas = document.getElementById('render-canvas') as HTMLCanvasElement;
  const fallback = document.getElementById('fallback') as HTMLParagraphElement;

  // Check WebGPU support
  if (!navigator.gpu) {
    fallback.style.display = 'block';
    canvas.style.display = 'none';
    console.warn('[WebGPU] Not available');
    return;
  }

  // Request adapter and device
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    fallback.style.display = 'block';
    canvas.style.display = 'none';
    console.warn('[WebGPU] No adapter found');
    return;
  }
  const device = await adapter.requestDevice();

  // Configure swap chain (now: getCurrentTexture)
  const context = canvas.getContext('webgpu') as GPUCanvasContext;
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format,
    alphaMode: 'premultiplied',
  });

  console.log('[WebGPU] Initialised', { format });

  // Render loop: clear to blue
  function frame() {
    const texture = context.getCurrentTexture();
    const view = texture.createView();
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view,
        clearValue: { r: 0.15, g: 0.15, b: 0.28, a: 1.0 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
    pass.end();
    device.queue.submit([encoder.finish()]);
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

main().catch(console.error);
