'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var get = require('../../helpers/get.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var tailwindMerge = require('../../helpers/tailwind-merge.cjs');
var provider = require('../../theme/provider.cjs');
var TableBodyContext = require('./TableBodyContext.cjs');
var TableContext = require('./TableContext.cjs');
var theme = require('./theme.cjs');

const TableBody = React.forwardRef((props, ref) => {
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme } = TableContext.useTableContext();
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.tableTheme.body, provider$1.theme?.table?.body, rootTheme?.body, props.theme],
    [get.get(provider$1.clearTheme, "table.body"), get.get(rootClearTheme, "body"), props.clearTheme],
    [get.get(provider$1.applyTheme, "table.body"), get.get(rootApplyTheme, "body"), props.applyTheme]
  );
  const { className, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.tableBody);
  return /* @__PURE__ */ jsxRuntime.jsx(
    TableBodyContext.TableBodyContext.Provider,
    {
      value: { theme: props.theme, clearTheme: props.clearTheme, applyTheme: props.applyTheme },
      children: /* @__PURE__ */ jsxRuntime.jsx("tbody", { ref, className: tailwindMerge.twMerge(theme$1.base, className), ...restProps })
    }
  );
});
TableBody.displayName = "TableBody";

exports.TableBody = TableBody;
//# sourceMappingURL=TableBody.cjs.map
