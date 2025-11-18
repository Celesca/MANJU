import { type ComponentPropsWithRef } from "react";
import type { ThemingProps } from "../../types";
import type { TableCellTheme } from "./TableCell";
export interface TableBodyTheme {
    base: string;
    cell: TableCellTheme;
}
export interface TableBodyProps extends ComponentPropsWithRef<"tbody">, ThemingProps<TableBodyTheme> {
}
export declare const TableBody: import("react").ForwardRefExoticComponent<Omit<TableBodyProps, "ref"> & import("react").RefAttributes<HTMLTableSectionElement>>;
