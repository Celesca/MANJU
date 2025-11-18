'use strict';

function mergeRefs(refs) {
  return (value) => {
    refs.forEach((ref) => {
      if (typeof ref === "function") {
        ref(value);
      } else if (ref != null) {
        const _ref = ref;
        _ref.current = value;
      }
    });
  };
}

exports.mergeRefs = mergeRefs;
//# sourceMappingURL=merge-refs.cjs.map
