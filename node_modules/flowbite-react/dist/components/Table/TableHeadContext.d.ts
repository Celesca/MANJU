import type { ThemingProps } from "../../types";
import type { TableHeadTheme } from "./TableHead";
export type TableHeadContextValue = ThemingProps<TableHeadTheme>;
export declare const TableHeadContext: import("react").Context<TableHeadContextValue | undefined>;
export declare function useTableHeadContext(): TableHeadContextValue;
