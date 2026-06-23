// Preload: Monkey-patch WeakMap for Bun + @vue/test-utils compatibility.
// @vue/test-utils calls WeakMap.prototype.set() with string keys,
// which Bun's native WeakMap rejects. This patch runs before any modules load.

const _set = WeakMap.prototype.set;
WeakMap.prototype.set = function (this: WeakMap<any, any>, key: any, value: any) {
  if (key !== null && (typeof key === 'object' || typeof key === 'function')) {
    return _set.call(this, key, value);
  }
  // Silently ignore non-object keys (matching old Node.js behavior)
  return this;
};
