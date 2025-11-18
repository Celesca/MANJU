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

const AvatarGroupCounter = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.avatarTheme.groupCounter, provider$1.theme?.avatar?.groupCounter, props.theme],
    [get.get(provider$1.clearTheme, "avatar.groupCounter"), props.clearTheme],
    [get.get(provider$1.applyTheme, "avatar.groupCounter"), props.applyTheme]
  );
  const { className, href, total, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.avatarGroupCounter);
  return /* @__PURE__ */ jsxRuntime.jsxs("a", { ref, href, className: tailwindMerge.twMerge(theme$1.base, className), ...restProps, children: [
    "+",
    total
  ] });
});
AvatarGroupCounter.displayName = "AvatarGroupCounter";

exports.AvatarGroupCounter = AvatarGroupCounter;
//# sourceMappingURL=AvatarGroupCounter.cjs.map
