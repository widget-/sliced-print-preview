declare module '*.glsl.js' {
  const src: string;
  export default src;
}

/** Float16Array — TC39 proposal, available in WebGPU-capable runtimes. */
interface Float16Array {
  readonly length: number;
  [index: number]: number;
  readonly BYTES_PER_ELEMENT: 2;
  buffer: ArrayBufferLike;
  byteOffset: number;
  byteLength: number;
  subarray(begin?: number, end?: number): Float16Array;
  slice(start?: number, end?: number): Float16Array;
}
declare var Float16Array: {
  readonly prototype: Float16Array;
  new (size: number): Float16Array;
  readonly BYTES_PER_ELEMENT: 2;
};

