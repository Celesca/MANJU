'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var get = require('../../helpers/get.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var tailwindMerge = require('../../helpers/tailwind-merge.cjs');
var provider = require('../../theme/provider.cjs');
var theme = require('./theme.cjs');

const AvatarGroup = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.avatarTheme.group, provider$1.theme?.avatar?.group, props.theme],
    [get.get(provider$1.clearTheme, "avatar"), props.clearTheme],
    [get.get(provider$1.applyTheme, "avatar"), props.applyTheme]
  );
  const { className, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.avatarGroup);
  return /* @__PURE__ */ jsxRuntime.jsx("div", { ref, "data-testid": "avatar-group-element", className: tailwindMerge.twMerge(theme$1.base, className), ...restProps });
});
AvatarGroup.displayName = "AvatarGroup";

exports.AvatarGroup = AvatarGroup;
//# sourceMappingURL=AvatarGroup.cjs.map
