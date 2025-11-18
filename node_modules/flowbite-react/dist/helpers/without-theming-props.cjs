'use strict';

function withoutThemingProps(props) {
  const { theme, clearTheme, applyTheme, ...rest } = props;
  return rest;
}

exports.withoutThemingProps = withoutThemingProps;
//# sourceMappingURL=without-theming-props.cjs.map
