'use strict';

var withoutThemingProps = require('./without-theming-props.cjs');

function resolveProps(props, providerProps) {
  let mergedProps = withoutThemingProps.withoutThemingProps(props);
  if (providerProps) {
    mergedProps = {
      ...providerProps,
      ...props
    };
  }
  return mergedProps;
}

exports.resolveProps = resolveProps;
//# sourceMappingURL=resolve-props.cjs.map
