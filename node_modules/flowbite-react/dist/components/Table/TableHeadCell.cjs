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
var TableHeadContext = require('./TableHeadContext.cjs');
var theme = require('./theme.cjs');

const TableHeadCell = React.forwardRef((props, ref) => {
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme } = TableContext.useTableContext();
  const { theme: headTheme, clearTheme: headClearTheme, applyTheme: headApplyTheme } = TableHeadContext.useTableHeadContext();
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.tableTheme.head.cell, provider$1.theme?.table?.head?.cell, rootTheme?.head?.cell, headTheme?.cell, props.theme],
    [
      get.get(provider$1.clearTheme, "table.head.cell"),
      get.get(rootClearTheme, "head.cell"),
      get.get(headClearTheme, "cell"),
      props.clearTheme
    ],
    [
      get.get(provider$1.applyTheme, "table.head.cell"),
      get.get(rootApplyTheme, "head.cell"),
      get.get(headApplyTheme, "cell"),
      props.applyTheme
    ]
  );
  const { className, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.tableHeadCell);
  return /* @__PURE__ */ jsxRuntime.jsx("th", { ref, className: tailwindMerge.twMerge(theme$1.base, className), ...restProps });
});
TableHeadCell.displayName = "TableHeadCell";

exports.TableHeadCell = TableHeadCell;
//# sourceMappingURL=TableHeadCell.cjs.map
