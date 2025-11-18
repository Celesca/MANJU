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

const TableRow = React.forwardRef((props, ref) => {
  const {
    theme: rootTheme,
    clearTheme: rootClearTheme,
    applyTheme: rootApplyTheme,
    hoverable,
    striped
  } = TableContext.useTableContext();
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.tableTheme.row, provider$1.theme?.table?.row, rootTheme?.row, props.theme],
    [get.get(provider$1.clearTheme, "table.row"), get.get(rootClearTheme, "row"), props.clearTheme],
    [get.get(provider$1.applyTheme, "table.row"), get.get(rootApplyTheme, "row"), props.applyTheme]
  );
  const { className, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.tableRow);
  return /* @__PURE__ */ jsxRuntime.jsx(
    "tr",
    {
      ref,
      "data-testid": "table-row-element",
      className: tailwindMerge.twMerge(theme$1.base, striped && theme$1.striped, hoverable && theme$1.hovered, className),
      ...restProps
    }
  );
});
TableRow.displayName = "TableRow";

exports.TableRow = TableRow;
//# sourceMappingURL=TableRow.cjs.map
