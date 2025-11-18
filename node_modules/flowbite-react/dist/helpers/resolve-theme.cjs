'use strict';

var deepmergeTs = require('deepmerge-ts');
var json = require('klona/json');
var React = require('react');
var index = require('../store/index.cjs');
var applyPrefix = require('./apply-prefix.cjs');
var applyPrefixV3 = require('./apply-prefix-v3.cjs');
var convertUtilitiesToV4 = require('./convert-utilities-to-v4.cjs');
var deepMerge = require('./deep-merge.cjs');
var isEqual = require('./is-equal.cjs');
var stripDark = require('./strip-dark.cjs');
var tailwindMerge = require('./tailwind-merge.cjs');

function useResolveTheme(...input) {
  return useStableMemo(() => resolveTheme(...input), input);
}
function useStableMemo(factory, dependencies) {
  const prevDepsRef = React.useRef();
  const prevResultRef = React.useRef();
  const hasChanged = !prevDepsRef.current || !isEqual.isEqual(prevDepsRef.current, dependencies);
  if (hasChanged) {
    prevDepsRef.current = dependencies;
    prevResultRef.current = factory();
  }
  return prevResultRef.current;
}
function resolveTheme([base, ...custom], clearThemeList, applyThemeList) {
  const dark = index.getDark();
  const prefix = index.getPrefix();
  const version = index.getVersion();
  const _custom = custom?.length ? custom?.filter((value) => value !== void 0) : void 0;
  const _clearThemeList = clearThemeList?.length ? clearThemeList?.filter((value) => value !== void 0) : void 0;
  const _applyThemeList = applyThemeList?.length ? applyThemeList?.filter((value) => value !== void 0) : void 0;
  const baseTheme = _clearThemeList?.length || dark === false || version === 4 || prefix ? json.klona(base) : base;
  if (_clearThemeList?.length) {
    const finalClearTheme = cloneWithValue(baseTheme, false);
    let run = false;
    for (const clearTheme of _clearThemeList) {
      if (clearTheme) {
        run = true;
      }
      patchClearTheme(finalClearTheme, clearTheme);
    }
    if (run) {
      runClearTheme(baseTheme, finalClearTheme);
    }
  }
  if (dark === false || version === 4 || prefix) {
    stringIterator(baseTheme, (value) => {
      if (dark === false) {
        value = stripDark.stripDark(value);
      }
      if (version === 4) {
        value = convertUtilitiesToV4.convertUtilitiesToV4(value);
      }
      if (prefix) {
        if (version === 3) {
          value = applyPrefixV3.applyPrefixV3(value, prefix);
        }
        if (version === 4) {
          value = applyPrefix.applyPrefix(value, prefix);
        }
      }
      return value;
    });
  }
  let theme = baseTheme;
  if (_custom?.length) {
    theme = deepMerge.deepMergeStrings(tailwindMerge.twMerge)(baseTheme, ..._custom);
  }
  if (_applyThemeList?.length && _custom?.length) {
    const finalApplyTheme = cloneWithValue(baseTheme, "merge");
    let run = false;
    for (const applyTheme of _applyThemeList) {
      if (applyTheme !== "merge") {
        run = true;
      }
      patchApplyTheme(finalApplyTheme, applyTheme);
    }
    if (run) {
      runApplyTheme(theme, deepmergeTs.deepmerge(baseTheme, ...custom), finalApplyTheme);
    }
  }
  return theme;
}
function patchClearTheme(base, clearTheme) {
  function iterate(base2, clearTheme2) {
    if (typeof clearTheme2 === "boolean") {
      if (typeof base2 === "object" && base2 !== null) {
        for (const key in base2) {
          base2[key] = iterate(base2[key], clearTheme2);
        }
      } else {
        return clearTheme2;
      }
    }
    if (typeof clearTheme2 === "object" && clearTheme2 !== null) {
      for (const key in clearTheme2) {
        base2[key] = iterate(base2[key], clearTheme2[key]);
      }
    }
    return base2;
  }
  iterate(base, clearTheme);
}
function patchApplyTheme(base, applyTheme) {
  function iterate(base2, applyTheme2) {
    if (typeof applyTheme2 === "string") {
      if (typeof base2 === "object" && base2 !== null) {
        for (const key in base2) {
          base2[key] = iterate(base2[key], applyTheme2);
        }
      } else {
        return applyTheme2;
      }
    }
    if (typeof applyTheme2 === "object" && applyTheme2 !== null) {
      for (const key in applyTheme2) {
        base2[key] = iterate(base2[key], applyTheme2[key]);
      }
    }
    return base2;
  }
  iterate(base, applyTheme);
}
function runClearTheme(base, clearTheme) {
  function iterate(base2, clearTheme2) {
    if (clearTheme2 === true) {
      if (typeof base2 === "object" && base2 !== null) {
        for (const key in base2) {
          base2[key] = iterate(base2[key], clearTheme2);
        }
      } else {
        return "";
      }
    }
    if (typeof clearTheme2 === "object" && clearTheme2 !== null) {
      for (const key in clearTheme2) {
        base2[key] = iterate(base2[key], clearTheme2[key]);
      }
    }
    return base2;
  }
  iterate(base, clearTheme);
}
function runApplyTheme(base, target, applyTheme) {
  function iterate(base2, target2, applyTheme2) {
    if (applyTheme2 === "replace") {
      if (typeof base2 === "object" && base2 !== null) {
        for (const key in base2) {
          base2[key] = iterate(base2[key], target2[key], applyTheme2);
        }
      } else {
        return target2;
      }
    }
    if (typeof applyTheme2 === "object" && applyTheme2 !== null) {
      for (const key in applyTheme2) {
        base2[key] = iterate(base2[key], target2[key], applyTheme2[key]);
      }
    }
    return base2;
  }
  iterate(base, target, applyTheme);
}
function stringIterator(input, callback) {
  function iterate(input2) {
    if (typeof input2 === "string") {
      return callback(input2);
    } else if (Array.isArray(input2)) {
      for (let i = 0; i < input2.length; i++) {
        input2[i] = iterate(input2[i]);
      }
    } else if (typeof input2 === "object" && input2 !== null) {
      for (const key in input2) {
        input2[key] = iterate(input2[key]);
      }
    }
    return input2;
  }
  iterate(input);
}
function cloneWithValue(input, value) {
  if (input === null || typeof input !== "object") {
    return value;
  }
  const clone = {};
  for (const key in input) {
    clone[key] = cloneWithValue(input[key], value);
  }
  return clone;
}

exports.resolveTheme = resolveTheme;
exports.useResolveTheme = useResolveTheme;
exports.useStableMemo = useStableMemo;
//# sourceMappingURL=resolve-theme.cjs.map
