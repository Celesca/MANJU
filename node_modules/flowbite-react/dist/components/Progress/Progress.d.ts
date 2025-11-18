import type { ComponentProps } from "react";
import type { DynamicStringEnumKeysOf, FlowbiteColors, FlowbiteSizes, ThemingProps } from "../../types";
export interface ProgressTheme {
    base: string;
    label: string;
    bar: string;
    color: ProgressColor;
    size: ProgressSizes;
}
export interface ProgressColor extends Pick<FlowbiteColors, "dark" | "blue" | "red" | "green" | "yellow" | "indigo" | "purple" | "cyan" | "gray" | "lime" | "pink" | "teal"> {
    [key: string]: string;
    default: string;
}
export interface ProgressSizes extends Pick<FlowbiteSizes, "sm" | "md" | "lg" | "xl"> {
    [key: string]: string;
}
export interface ProgressProps extends ComponentProps<"div">, ThemingProps<ProgressTheme> {
    color?: DynamicStringEnumKeysOf<ProgressColor>;
    labelProgress?: boolean;
    labelText?: boolean;
    progress: number;
    progressLabelPosition?: "inside" | "outside";
    size?: DynamicStringEnumKeysOf<ProgressSizes>;
    textLabel?: string;
    textLabelPosition?: "inside" | "outside";
}
export declare const Progress: import("react").ForwardRefExoticComponent<Omit<ProgressProps, "ref"> & import("react").RefAttributes<HTMLDivElement>>;
