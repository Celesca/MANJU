import { extendTailwindMerge as extendTailwindMerge$1 } from 'tailwind-merge-v2';
import { extendTailwindMerge } from 'tailwind-merge-v3';
import { getPrefix, getVersion } from '../store/index.js';

const cache = /* @__PURE__ */ new Map();
function twMerge(...classLists) {
  const prefix = getPrefix();
  const version = getVersion();
  const cacheKey = `${prefix}.${version}`;
  const cacheValue = cache.get(cacheKey);
  if (cacheValue) {
    return cacheValue(...classLists);
  }
  const twMergeFn = (version === 3 ? extendTailwindMerge$1 : extendTailwindMerge)({
    extend: {
      classGroups: {
        "bg-image": ["bg-arrow-down-icon", "bg-check-icon", "bg-dash-icon", "bg-dot-icon"],
        shadow: ["shadow-sm-light"]
      }
    },
    prefix
  });
  cache.set(cacheKey, twMergeFn);
  return twMergeFn(...classLists);
}

export { twMerge };
//# sourceMappingURL=tailwind-merge.js.map
