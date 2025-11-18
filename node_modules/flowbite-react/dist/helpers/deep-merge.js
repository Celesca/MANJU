import { deepmergeCustom } from 'deepmerge-ts';

function deepMergeStrings(merge) {
  return deepmergeCustom({
    mergeOthers: (values, utils) => {
      if (values.some((value) => typeof value === "string")) {
        const strings = values.filter((value) => typeof value === "string");
        const stringMap = /* @__PURE__ */ new Set();
        const uniqueStrings = [];
        for (const string of strings) {
          const parts = [...new Set(string.split(/\s+/))];
          uniqueStrings.push(parts.filter((part) => !stringMap.has(part)).join(" "));
          for (const part of parts) {
            stringMap.add(part);
          }
        }
        return merge(uniqueStrings);
      }
      return utils.actions.defaultMerge;
    }
  });
}

export { deepMergeStrings };
//# sourceMappingURL=deep-merge.js.map
