'use strict';

function isEqual(a, b) {
  if (a === b) {
    return true;
  }
  if (a && b && typeof a === "object" && typeof b === "object") {
    if (a.constructor !== b.constructor) {
      return false;
    }
    if (Array.isArray(a)) {
      if (a.length !== b.length) {
        return false;
      }
      return a.every((item, index) => isEqual(item, b[index]));
    }
    if (a.constructor === RegExp) {
      return a.source === b.source && a.flags === b.flags;
    }
    if (a.valueOf !== Object.prototype.valueOf) {
      return a.valueOf() === b.valueOf();
    }
    if (a.toString !== Object.prototype.toString) {
      return a.toString() === b.toString();
    }
    const aKeys = Object.keys(a);
    if (aKeys.length !== Object.keys(b).length) {
      return false;
    }
    return aKeys.every((key) => Object.prototype.hasOwnProperty.call(b, key) && isEqual(a[key], b[key]));
  }
  return a !== a && b !== b;
}

exports.isEqual = isEqual;
//# sourceMappingURL=is-equal.cjs.map
