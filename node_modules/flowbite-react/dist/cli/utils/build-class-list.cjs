'use strict';

var applyPrefix = require('../../helpers/apply-prefix.cjs');
var applyPrefixV3 = require('../../helpers/apply-prefix-v3.cjs');
var convertUtilitiesToV4 = require('../../helpers/convert-utilities-to-v4.cjs');
var stripDark = require('../../helpers/strip-dark.cjs');
var classList = require('../../metadata/class-list.cjs');
var dependencyList = require('../../metadata/dependency-list.cjs');

function buildClassList({
  components,
  dark,
  prefix,
  version
}) {
  let classList$1 = [];
  if (components.includes("*")) {
    classList$1 = [...new Set(Object.values(classList.CLASS_LIST_MAP).flat())];
  } else {
    let resolveDependencies2 = function(name) {
      if (visited.has(name)) {
        return;
      }
      visited.add(name);
      if (name in dependencyList.DEPENDENCY_LIST_MAP) {
        resolvedComponents.add(name);
        for (const dependency of dependencyList.DEPENDENCY_LIST_MAP[name]) {
          if (dependency in dependencyList.DEPENDENCY_LIST_MAP) {
            resolvedComponents.add(dependency);
            resolveDependencies2(dependency);
          }
        }
      }
    };
    const resolvedComponents = /* @__PURE__ */ new Set();
    const visited = /* @__PURE__ */ new Set();
    for (const name of components) {
      resolveDependencies2(name);
    }
    classList$1 = [
      ...new Set(
        [...resolvedComponents].flatMap((name) => {
          const classListKey = classList.COMPONENT_TO_CLASS_LIST_MAP[name];
          const resolvedClassList = classList.CLASS_LIST_MAP[classListKey];
          return resolvedClassList || [];
        })
      )
    ];
  }
  if (dark === false) {
    classList$1 = classList$1.map(stripDark.stripDark).filter(Boolean);
  }
  if (version === 4) {
    classList$1 = classList$1.map(convertUtilitiesToV4.convertUtilitiesToV4);
  }
  if (prefix?.trim()) {
    classList$1 = classList$1.map(
      (className) => version === 3 ? applyPrefixV3.applyPrefixV3(className, prefix) : applyPrefix.applyPrefix(className, prefix)
    );
  }
  return classList$1.sort();
}

exports.buildClassList = buildClassList;
//# sourceMappingURL=build-class-list.cjs.map
