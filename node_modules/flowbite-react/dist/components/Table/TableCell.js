'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { useTableBodyContext } from './TableBodyContext.js';
import { useTableContext } from './TableContext.js';
import { tableTheme } from './theme.js';

const TableCell = forwardRef((props, ref) => {
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme } = useTableContext();
  const { theme: bodyTheme, clearTheme: bodyClearTheme, applyTheme: bodyApplyTheme } = useTableBodyContext();
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [tableTheme.body.cell, provider.theme?.table?.body?.cell, rootTheme?.body?.cell, bodyTheme?.cell, props.theme],
    [
      get(provider.clearTheme, "table.body.cell"),
      get(rootClearTheme, "body.cell"),
      get(bodyClearTheme, "cell"),
      props.clearTheme
    ],
    [
      get(provider.applyTheme, "table.body.cell"),
      get(rootApplyTheme, "body.cell"),
      get(bodyApplyTheme, "cell"),
      props.applyTheme
    ]
  );
  const { className, ...restProps } = resolveProps(props, provider.props?.tableCell);
  return /* @__PURE__ */ jsx("td", { ref, className: twMerge(theme.base, className), ...restProps });
});
TableCell.displayName = "TableCell";

export { TableCell };
//# sourceMappingURL=TableCell.js.map
