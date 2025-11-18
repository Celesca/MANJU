import { type ComponentPropsWithRef } from "react";
import type { ThemingProps } from "../../types";
export interface TableCellTheme {
    base: string;
}
export interface TableCellProps extends ComponentPropsWithRef<"td">, ThemingProps<TableCellTheme> {
}
export declare const TableCell: import("react").ForwardRefExoticComponent<Omit<TableCellProps, "ref"> & import("react").RefAttributes<HTMLTableCellElement>>;
