import { type ComponentPropsWithRef } from "react";
import type { ThemingProps } from "../../types";
import type { TableHeadCellTheme } from "./TableHeadCell";
export interface TableHeadTheme {
    base: string;
    cell: TableHeadCellTheme;
}
export interface TableHeadProps extends ComponentPropsWithRef<"thead">, ThemingProps<TableHeadTheme> {
}
export declare const TableHead: import("react").ForwardRefExoticComponent<Omit<TableHeadProps, "ref"> & import("react").RefAttributes<HTMLTableSectionElement>>;
