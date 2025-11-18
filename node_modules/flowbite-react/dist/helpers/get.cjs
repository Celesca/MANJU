'use strict';

function get(input, path) {
  const keys = path.split(".");
  let result = input;
  for (const key of keys) {
    if (typeof result === "boolean" || typeof result === "string") {
      return result;
    }
    if (result == null || typeof result !== "object") {
      return void 0;
    }
    result = result[key];
  }
  return result;
}

exports.get = get;
//# sourceMappingURL=get.cjs.map
