const cache = /* @__PURE__ */ new Map();
function stripDark(classNames) {
  if (classNames === void 0 || classNames === null) {
    return classNames;
  }
  if (!classNames.trim().length) {
    return classNames;
  }
  classNames = classNames.trim();
  const cacheKey = classNames;
  const cacheValue = cache.get(cacheKey);
  if (cacheValue) {
    return cacheValue;
  }
  const result = classNames.split(/\s+/).filter((className) => !className.includes("dark:")).join(" ");
  cache.set(cacheKey, result);
  return result;
}

export { stripDark };
//# sourceMappingURL=strip-dark.js.map
