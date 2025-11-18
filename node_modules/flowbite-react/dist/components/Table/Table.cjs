'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var get = require('../../helpers/get.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var tailwindMerge = require('../../helpers/tailwind-merge.cjs');
var provider = require('../../theme/provider.cjs');
var TableContext = require('./TableContext.cjs');
var theme = require('./theme.cjs');

const Table = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.tableTheme, provider$1.theme?.table, props.theme],
    [get.get(provider$1.clearTheme, "table"), props.clearTheme],
    [get.get(provider$1.applyTheme, "table"), props.applyTheme]
  );
  const { className, striped, hoverable, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.table);
  return /* @__PURE__ */ jsxRuntime.jsx("div", { "data-testid": "table-element", className: tailwindMerge.twMerge(theme$1.root.wrapper), children: /* @__PURE__ */ jsxRuntime.jsxs(
    TableContext.TableContext.Provider,
    {
      value: { theme: props.theme, clearTheme: props.clearTheme, applyTheme: props.applyTheme, striped, hoverable },
      children: [
        /* @__PURE__ */ jsxRuntime.jsx("div", { className: tailwindMerge.twMerge(theme$1.root.shadow, className) }),
        /* @__PURE__ */ jsxRuntime.jsx("table", { ref, className: tailwindMerge.twMerge(theme$1.root.base, className), ...restProps })
      ]
    }
  ) });
});
Table.displayName = "Table";

exports.Table = Table;
//# sourceMappingURL=Table.cjs.map
