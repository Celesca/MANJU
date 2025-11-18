import { type ComponentProps, type FC } from "react";
import type { FlowbiteSizes, ThemingProps } from "../../types";
export interface RatingStarTheme {
    empty: string;
    filled: string;
    sizes: RatingStarSizes;
}
export interface RatingStarSizes extends Pick<FlowbiteSizes, "sm" | "md" | "lg"> {
    [key: string]: string;
}
export interface RatingStarProps extends ComponentProps<"svg">, ThemingProps<RatingStarTheme> {
    filled?: boolean;
    starIcon?: FC<ComponentProps<"svg">>;
}
export declare const RatingStar: import("react").ForwardRefExoticComponent<Omit<RatingStarProps, "ref"> & import("react").RefAttributes<SVGSVGElement>>;
