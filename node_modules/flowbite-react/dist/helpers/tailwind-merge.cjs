'use strict';

var tailwindMergeV2 = require('tailwind-merge-v2');
var tailwindMergeV3 = require('tailwind-merge-v3');
var index = require('../store/index.cjs');

const cache = /* @__PURE__ */ new Map();
function twMerge(...classLists) {
  const prefix = index.getPrefix();
  const version = index.getVersion();
  const cacheKey = `${prefix}.${version}`;
  const cacheValue = cache.get(cacheKey);
  if (cacheValue) {
    return cacheValue(...classLists);
  }
  const twMergeFn = (version === 3 ? tailwindMergeV2.extendTailwindMerge : tailwindMergeV3.extendTailwindMerge)({
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

exports.twMerge = twMerge;
//# sourceMappingURL=tailwind-merge.cjs.map
