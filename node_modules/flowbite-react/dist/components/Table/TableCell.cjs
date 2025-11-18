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

const TableCell = React.forwardRef((props, ref) => {
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme } = TableContext.useTableContext();
  const { theme: bodyTheme, clearTheme: bodyClearTheme, applyTheme: bodyApplyTheme } = TableBodyContext.useTableBodyContext();
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.tableTheme.body.cell, provider$1.theme?.table?.body?.cell, rootTheme?.body?.cell, bodyTheme?.cell, props.theme],
    [
      get.get(provider$1.clearTheme, "table.body.cell"),
      get.get(rootClearTheme, "body.cell"),
      get.get(bodyClearTheme, "cell"),
      props.clearTheme
    ],
    [
      get.get(provider$1.applyTheme, "table.body.cell"),
      get.get(rootApplyTheme, "body.cell"),
      get.get(bodyApplyTheme, "cell"),
      props.applyTheme
    ]
  );
  const { className, ...restProps } = resolveProps.resolveProps(props, provider$1.props?.tableCell);
  return /* @__PURE__ */ jsxRuntime.jsx("td", { ref, className: tailwindMerge.twMerge(theme$1.base, className), ...restProps });
});
TableCell.displayName = "TableCell";

exports.TableCell = TableCell;
//# sourceMappingURL=TableCell.cjs.map
