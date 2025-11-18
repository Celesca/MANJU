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

const List = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.listTheme.root, provider$1.theme?.list?.root, props.theme],
    [get.get(provider$1.clearTheme, "list.root"), props.clearTheme],
    [get.get(provider$1.applyTheme, "list.root"), props.applyTheme]
  );
  const { className, horizontal, nested, ordered, unstyled, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.list);
  const Component = ordered ? "ol" : "ul";
  return /* @__PURE__ */ jsxRuntime.jsx(
    Component,
    {
      ref,
      className: tailwindMerge.twMerge(
        theme$1.base,
        theme$1.ordered[ordered ? "on" : "off"],
        unstyled && theme$1.unstyled,
        nested && theme$1.nested,
        horizontal && theme$1.horizontal,
        className
      ),
      ...restProps
    }
  );
});
List.displayName = "List";

exports.List = List;
//# sourceMappingURL=List.cjs.map
