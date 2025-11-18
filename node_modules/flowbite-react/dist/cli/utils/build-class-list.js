import { applyPrefix } from '../../helpers/apply-prefix.js';
import { applyPrefixV3 } from '../../helpers/apply-prefix-v3.js';
import { convertUtilitiesToV4 } from '../../helpers/convert-utilities-to-v4.js';
import { stripDark } from '../../helpers/strip-dark.js';
import { CLASS_LIST_MAP, COMPONENT_TO_CLASS_LIST_MAP } from '../../metadata/class-list.js';
import { DEPENDENCY_LIST_MAP } from '../../metadata/dependency-list.js';

function buildClassList({
  components,
  dark,
  prefix,
  version
}) {
  let classList = [];
  if (components.includes("*")) {
    classList = [...new Set(Object.values(CLASS_LIST_MAP).flat())];
  } else {
    let resolveDependencies2 = function(name) {
      if (visited.has(name)) {
        return;
      }
      visited.add(name);
      if (name in DEPENDENCY_LIST_MAP) {
        resolvedComponents.add(name);
        for (const dependency of DEPENDENCY_LIST_MAP[name]) {
          if (dependency in DEPENDENCY_LIST_MAP) {
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
    classList = [
      ...new Set(
        [...resolvedComponents].flatMap((name) => {
          const classListKey = COMPONENT_TO_CLASS_LIST_MAP[name];
          const resolvedClassList = CLASS_LIST_MAP[classListKey];
          return resolvedClassList || [];
        })
      )
    ];
  }
  if (dark === false) {
    classList = classList.map(stripDark).filter(Boolean);
  }
  if (version === 4) {
    classList = classList.map(convertUtilitiesToV4);
  }
  if (prefix?.trim()) {
    classList = classList.map(
      (className) => version === 3 ? applyPrefixV3(className, prefix) : applyPrefix(className, prefix)
    );
  }
  return classList.sort();
}

export { buildClassList };
//# sourceMappingURL=build-class-list.js.map
