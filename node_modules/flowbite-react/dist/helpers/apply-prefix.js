const cache = /* @__PURE__ */ new Map();
function applyPrefix(classNames, prefix) {
  if (!classNames.trim().length || !prefix.trim().length) {
    return classNames;
  }
  classNames = classNames.trim();
  prefix = prefix.trim();
  const cacheKey = `${classNames}.${prefix}`;
  const cacheValue = cache.get(cacheKey);
  if (cacheValue) {
    return cacheValue;
  }
  const result = classNames.split(/\s+/).map((className) => {
    className = className.trim();
    if (!className.length || className.startsWith(prefix)) {
      return className;
    }
    return `${prefix}:${className}`;
  }).join(" ");
  cache.set(cacheKey, result);
  return result;
}

export { applyPrefix };
//# sourceMappingURL=apply-prefix.js.map
