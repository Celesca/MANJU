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

export { mergeRefs };
//# sourceMappingURL=merge-refs.js.map
