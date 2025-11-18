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

const Kbd = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.kbdTheme, provider$1.theme?.kbd, props.theme],
    [get.get(provider$1.clearTheme, "kbd"), props.clearTheme],
    [get.get(provider$1.applyTheme, "kbd"), props.applyTheme]
  );
  const { children, className, icon: Icon, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.hr);
  return /* @__PURE__ */ jsxRuntime.jsxs("span", { ref, className: tailwindMerge.twMerge(theme$1.root.base, className), "data-testid": "flowbite-kbd", ...restProps, children: [
    Icon && /* @__PURE__ */ jsxRuntime.jsx(Icon, { className: theme$1.root.icon, "data-testid": "flowbite-kbd-icon" }),
    children
  ] });
});
Kbd.displayName = "Kbd";

exports.Kbd = Kbd;
//# sourceMappingURL=Kbd.cjs.map
